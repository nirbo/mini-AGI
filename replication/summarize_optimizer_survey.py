"""CPU-only evidence assembly for optimizer research; no training changes."""

import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile


def main():
    parser_root = Path("/home/nir/ml-tools/mini-AGI/runs/performance/optimizers")
    paths = sorted(p for p in parser_root.glob("*.json") if not p.name.endswith(".power.json") and p.name != "SHA256SUMS.json")
    results = {p.stem: json.loads(p.read_text()) for p in paths}
    for name in results:
        source = Path(f"/tmp/minagi-opt-{name}.log")
        if source.exists():
            shutil.copyfile(source, parser_root / f"{name}.log")
    shutil.copyfile("/tmp/minagi-optimizer-survey-suite.log", parser_root / "suite.log")
    groups = defaultdict(list)
    for name, result in results.items():
        if name.startswith("latency-"):
            assert result["status"] == "passed"
            cfg = result["configuration"]
            groups[(cfg["layout"], cfg["optimizer"])].append(result["result"])
    summary = []
    for (layout, optimizer), rows in sorted(groups.items()):
        duration = sum(r["end_monotonic"] - r["start_monotonic"] for r in rows)
        energy = sum(r["power"]["estimated_board_energy_j"] for r in rows)
        summary.append({"layout": layout, "optimizer": optimizer, "runs": len(rows),
                        "median_step_ms_range": [min(r["median_step_seconds"] for r in rows) * 1000,
                                                 max(r["median_step_seconds"] for r in rows) * 1000],
                        "state_bytes_per_parameter_range": [min(r["state_bytes_per_parameter"] for r in rows),
                                                             max(r["state_bytes_per_parameter"] for r in rows)],
                        "average_board_power_w": energy / duration,
                        "peak_sampled_board_power_w": max(r["power"]["peak_sampled_board_power_w"] for r in rows),
                        "board_joules_per_optimizer_step": energy / sum(r["iterations"] for r in rows),
                        "peak_allocated_gib": max(r["peak_allocated_gib"] for r in rows)})
    original = ast.parse(subprocess.check_output(["git", "show", "33f5591:minagi/precision.py"], text=True))
    current = ast.parse(Path("minagi/precision.py").read_text())
    original.body, current.body = original.body[1:], current.body[1:]
    executable_unchanged = ast.dump(original, include_attributes=False) == ast.dump(current, include_attributes=False)
    assert executable_unchanged
    assert "Ran 81 tests" in (parser_root / "suite.log").read_text()
    assert "\nOK\n" in (parser_root / "suite.log").read_text()
    assert results["contracts-gefen-patched"]["status"] == "passed"
    assert results["real-gradients"]["status"] == "passed"
    report = {"optimizer_summary": summary, "results": results,
              "precision_executable_ast_unchanged": executable_unchanged, "tests_passed": 81,
              "disposition": "Research only; no training optimizer replacement. Existing pager cannot preserve candidate states.",
              "limits": ["Synthetic optimizer-only timings cannot establish end-to-end speed or time to a target validation loss.",
                         "Real-gradient state sizes use one checkpoint batch, not a convergence experiment or guaranteed compression ratio.",
                         "The Gefen build uses a documented opt-in same-major CUDA minor-version patch; no kernel arithmetic is modified.",
                         "No model checkpoint is written. Current production weights, moments, optimizer and paging behavior remain unchanged."]}
    retained = sorted(p for p in parser_root.iterdir() if p.is_file() and p.suffix in (".json", ".log") and p.name != "SHA256SUMS.json")
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in retained}
    manifest_path = parser_root / "SHA256SUMS.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    archive = Path("replication/results/optimizer_survey_evidence.tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        for path in [*retained, manifest_path]:
            tar.add(path, arcname=path.name)
    with tarfile.open(archive) as tar:
        assert all(hashlib.sha256(tar.extractfile(name).read()).hexdigest() == digest for name, digest in manifest.items())
    report["evidence"] = {"path": str(archive), "files": len(retained),
                          "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}
    Path("replication/results/optimizer_survey.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
