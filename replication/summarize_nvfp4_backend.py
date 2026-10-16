"""Assemble retained qualification evidence; CPU/file operations only."""

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tarfile

from replication.gpu_monitor import summarize_samples


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs/performance/native-nvfp4"))
    parser.add_argument("--out", type=Path, default=Path("replication/results/native_nvfp4_backend.json"))
    args = parser.parse_args()
    root = args.runs
    args.out.parent.mkdir(parents=True, exist_ok=True)
    log_names = ["suite-final", "gradients-final", "learning-final", "benchmark-final"]
    log_names += [f"model-{cp}-{i + 1}-{b}" for cp in ("on", "off")
                  for i, b in enumerate(("sonic", "nvfp4", "nvfp4", "sonic"))]
    for name in log_names:
        source, target = Path(f"/tmp/minagi-{name}.log"), root / f"{name}.log"
        if source.exists():
            shutil.copyfile(source, target)
        elif not target.exists():
            raise FileNotFoundError(f"missing retained qualification log: {target}")
    for name in ("native-nvfp4-tests", "native-nvfp4-tests2", "native-integration", "native-suite",
                 "native-gradients", "native-learning", "native-layer", "native-model-pilot",
                 "native-equivalence", "native-equivalence-final"):
        log = Path(f"/tmp/minagi-{name}.log")
        if log.exists():
            shutil.copyfile(log, root / f"{name}.log")

    verification = {mode: read(root / f"{mode}-final.json") for mode in
                    ("gradients", "learning", "benchmark", "batching-equivalence")}
    assert all(result["status"] == "passed" for result in verification.values())
    suite = (root / "suite-final.log").read_text()
    assert re.search(r"\nOK\n", suite)
    tests = int(re.search(r"Ran (\d+) tests", suite)[1])
    models = []
    for name in log_names[4:]:
        result = read(root / f"{name}.json")
        power = read(root / f"{name}.power.json")
        start, end = result["timing_interval_unix"]
        samples = [{**s, "monotonic": s["unix_time"]} for s in power["samples"]]
        cfg = result["configuration"]
        measured = summarize_samples(samples, start, end, new_tokens=cfg["steps"] * cfg["stride"])
        measured["start_unix"], measured["end_unix"] = measured.pop("start_monotonic"), measured.pop("end_monotonic")
        assert all(math.isfinite(n) for n in result["losses"] + result["gradient_norms"])
        models.append({"name": name, "backend": cfg["expert_backend"],
                       "checkpoint": not cfg["no_checkpoint"], "power": measured,
                       "median_step_seconds": result["median_step_seconds"],
                       "step_seconds": result["step_seconds"], "peak_allocated_gib": result["peak_allocated_gib"],
                       "losses": result["losses"], "gradient_norms": result["gradient_norms"],
                       "git_commit": result["git_commit"], "manifest_sha256": result["manifest_sha256"]})
    aggregates = {}
    for checkpoint in (True, False):
        group = {}
        for backend in ("sonic", "nvfp4"):
            rows = [r for r in models if r["checkpoint"] == checkpoint and r["backend"] == backend]
            duration = sum(r["power"]["duration_seconds"] for r in rows)
            energy = sum(r["power"]["estimated_board_energy_j"] for r in rows)
            tokens = sum(r["power"]["new_tokens"] for r in rows)
            group[backend] = {"runs": len(rows), "new_tokens_per_second": tokens / duration,
                              "board_joules_per_new_token": energy / tokens,
                              "average_board_power_w": energy / duration,
                              "peak_sampled_board_power_w": max(r["power"]["peak_sampled_board_power_w"] for r in rows),
                              "peak_allocated_gib_range": [min(r["peak_allocated_gib"] for r in rows),
                                                           max(r["peak_allocated_gib"] for r in rows)]}
        group["nvfp4_throughput_ratio_to_sonic"] = group["nvfp4"]["new_tokens_per_second"] / group["sonic"]["new_tokens_per_second"]
        group["nvfp4_energy_ratio_to_sonic"] = group["nvfp4"]["board_joules_per_new_token"] / group["sonic"]["board_joules_per_new_token"]
        aggregates["checkpoint_on" if checkpoint else "checkpoint_off"] = group

    def functions(source):
        return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(source).body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    before = functions(subprocess.check_output(["git", "show", "33f5591:minagi/expert_compute.py"], text=True))
    after = functions(Path("minagi/expert_compute.py").read_text())
    unchanged = {name: before[name] == after[name] for name in ("_reference", "_few", "_sonic")}
    assert all(unchanged.values())
    paths = [Path(p) for p in ("minagi/pool.py", "minagi/paged.py", "minagi/precision.py")]
    assert not subprocess.check_output(["git", "diff", "33f5591", "--", *map(str, paths)])
    source_paths = [Path(p) for p in ("minagi/nvfp4.py", "minagi/expert_compute.py",
                    "replication/qualify_nvfp4_backend.py", "replication/profile_training.py",
                    "tests/test_nvfp4_backend.py", "tests/test_causality.py")]
    source_hashes = {str(p): sha(p) for p in source_paths}
    assert len({r["git_commit"] for r in models}) == 1
    for mode in ("gradients", "learning", "benchmark"):
        for name, digest in verification[mode]["sources_sha256"].items():
            assert sha(Path(name)) == digest, (mode, name)
    trace = root / "model-pilot.trace.json"
    report = {"disposition": "Native arithmetic qualified on tested inputs; practical training backend not qualified. Feature branch remains unmerged.",
              "tested_source_commit": models[0]["git_commit"], "source_sha256": source_hashes,
              "tests_passed": tests, "unchanged_reference_function_ast": unchanged,
              "pool_paging_precision_unchanged": True, "qualification": verification,
              "model_runs": models, "model_aggregates": aggregates,
              "eager_layer_baseline": read(root / "layer-eager.json"),
              "eager_model_profile": read(root / "model-pilot.json")["profile"],
              "local_only_trace": {"path": str(trace.resolve()), "bytes": trace.stat().st_size, "sha256": sha(trace)},
              "limits": ["One RTX 5090, one text stream, fixed depth 24, fresh optimizer, eight measured updates after two warmups per run.",
                         "ABBA comparisons per checkpoint setting; warm caches. Peak allocated VRAM includes warmup, not only the timed interval.",
                         "Short windows and averaged NVML sensors limit power precision; samples include desktop activity and exclude CPU energy.",
                         "Native FP4 changes the forward computation and stochastic gradient trajectory. No matched language-model convergence/quality claim.",
                         "The regression control is untuned; its failure to beat BF16 plus PTQ does not prove NVFP4 training cannot improve with a different recipe.",
                         "The eager profile predates arithmetic-preserving batching; instrumented timing is not an occupancy measurement."]}
    (root / "source-sha256.json").write_text(json.dumps(source_hashes, indent=2) + "\n")
    files = sorted(p for p in root.iterdir() if p.is_file() and p.suffix in (".json", ".log")
                   and p != trace and p.name != "SHA256SUMS.json")
    manifest = {p.name: sha(p) for p in files}
    (root / "SHA256SUMS.json").write_text(json.dumps(manifest, indent=2) + "\n")
    archive = args.out.with_name("native_nvfp4_backend_evidence.tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        for path in [*files, root / "SHA256SUMS.json"]:
            tar.add(path, arcname=path.name)
    report["evidence_archive"] = {"path": str(archive), "bytes": archive.stat().st_size,
                                   "sha256": sha(archive), "files_excluding_manifest": len(files)}
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    with tarfile.open(archive) as tar:
        for name, expected in manifest.items():
            assert hashlib.sha256(tar.extractfile(name).read()).hexdigest() == expected, name
    print(json.dumps({"report": str(args.out), "tests": tests, "model_aggregates": aggregates,
                      "evidence": report["evidence_archive"]}, indent=2))


if __name__ == "__main__":
    main()
