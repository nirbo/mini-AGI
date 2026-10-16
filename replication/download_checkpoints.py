"""Download revision-pinned upstream checkpoints, verifying every file.

Run with the repository venv, with huggingface_hub installed there.
Existing verified files are reused, so interrupted downloads can be retried.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
from urllib.request import urlopen


SNAPSHOTS = {
    "undertrained": ("Volotat/mini-AGI-undertrained",
                     "5906e846242d80a487f7bb685d88f4751f2f88ec"),
    "cl-replication": ("Volotat/mini-AGI-cl-replication-weights",
                       "669afac8d059a238374ca2071ff6de92d5961d08"),
}


def digest(path, git_blob=False):
    h = hashlib.sha1() if git_blob else hashlib.sha256()
    if git_blob:
        h.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download(root, name, workers=4):
    repo, revision = SNAPSHOTS[name]
    with urlopen(f"https://huggingface.co/api/models/{repo}/revision/"
                 f"{revision}?blobs=true", timeout=60) as r:
        info = json.load(r)
    if info["sha"] != revision:
        raise ValueError("server returned a different revision")
    files = info["siblings"]
    dest = root / name
    dest.mkdir(parents=True, exist_ok=True)
    total = sum(f["size"] for f in files)
    print(f"{name}: {len(files)} files, {total / 2**30:.3f} GiB -> {dest}",
          flush=True)
    subprocess.run([str(Path(sys.executable).parent / "hf"), "download", repo,
                    "--revision", revision, "--local-dir", str(dest),
                    "--max-workers", str(workers), "--quiet"], check=True)

    def fetch(entry):
        rel = PurePosixPath(entry["rfilename"])
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"unsafe remote path: {rel}")
        target = dest.joinpath(*rel.parts)
        lfs = entry.get("lfs")
        expected = lfs["sha256"] if lfs else entry["blobId"]

        def valid(path):
            return (path.is_file() and path.stat().st_size == entry["size"]
                    and digest(path, git_blob=not lfs) == expected)

        if valid(target):
            return
        raise ValueError(f"size/hash verification failed: {target}")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for i, _ in enumerate(executor.map(fetch, files), 1):
            if i % 25 == 0 or i == len(files):
                print(f"{name}: verified {i}/{len(files)} files", flush=True)
    provenance = {"repo": repo, "revision": revision, "bytes": total,
                  "files": files, "verification": "SHA-256 for LFS; Git blob SHA-1 otherwise"}
    (dest / "download-provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n")
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--snapshot", choices=["all", *SNAPSHOTS], default="all")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    for snapshot in SNAPSHOTS if args.snapshot == "all" else [args.snapshot]:
        download(args.root.expanduser(), snapshot, args.workers)
