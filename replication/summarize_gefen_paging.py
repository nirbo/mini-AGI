"""Assemble local Gefen integration/learning evidence without GPU work."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import tarfile

from replication.gpu_monitor import summarize_samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("/home/nir/ml-tools/mini-AGI/runs/performance/gefen-paging"))
    args = parser.parse_args()
    root = args.runs
    for path in Path("/tmp").glob("minagi-gefen-*.log"):
        shutil.copyfile(path, root / path.name)
    results = {}
    for path in sorted(root.glob("*-lr*.json")):
        if path.name.endswith(".power.json"):
            continue
        result = json.loads(path.read_text())
        assert result["status"] == "passed", path
        external = path.with_suffix(".external.power.json")
        if external.exists():
            power = json.loads(external.read_text())
            assert power["returncode"] == 0 and not power["errors"]
            result["qualified_power"] = summarize_samples(power["samples"], result["training_start"],
                                                         result["training_end"],
                                                         result["configuration"]["steps"] * result["configuration"]["context"])
            result["power_source"] = external.name
        else:
            result["qualified_power"] = result["training_power"]
            result["power_source"] = path.with_suffix(".power.json").name
        results[path.name] = result
    assert len(results) == 20
    grid = [d for d in results.values() if d["configuration"]["seed"] == 42]
    assert len(grid) == 12
    best = {opt: min((d for d in grid if d["configuration"]["optimizer"] == opt),
                     key=lambda d: d["final_tune"]["mean_ce"])["configuration"]["lr"]
            for opt in ("adamw", "gefen")}
    assert best == {"adamw": 1e-4, "gefen": 1e-4}
    comparison = []
    for steps, seeds in ((64, (42, 43, 44)), (256, (43, 44))):
        for opt in ("adamw", "gefen"):
            rows = [d for d in results.values() if d["configuration"]["optimizer"] == opt
                    and d["configuration"]["lr"] == best[opt]
                    and d["configuration"]["steps"] == steps]
            assert sorted(d["configuration"]["seed"] for d in rows) == list(seeds)
            duration = sum(d["qualified_power"]["duration_seconds"] for d in rows)
            energy = sum(d["qualified_power"]["estimated_board_energy_j"] for d in rows)
            tokens = sum(d["qualified_power"]["new_tokens"] for d in rows)
            comparison.append({"optimizer": opt, "steps": steps, "seeds": list(seeds),
                               "mean_test_ce": statistics.mean(d["final_test"]["mean_ce"] for d in rows),
                               "mean_tune_ce": statistics.mean(d["final_tune"]["mean_ce"] for d in rows),
                               "mean_training_seconds": duration / len(rows),
                               "new_tokens_per_second": tokens / duration,
                               "average_board_power_w": energy / duration,
                               "peak_sampled_board_power_w": max(d["qualified_power"]["peak_sampled_board_power_w"] for d in rows),
                               "board_joules_per_new_token": energy / tokens,
                               "peak_allocated_gib": max(d["peak_allocated_gib"] for d in rows),
                               "mean_last_16_step_seconds": statistics.mean(t["seconds"] for d in rows for t in d["trace"][-16:]),
                               "max_power_sample_gap_seconds": max(d["qualified_power"]["max_sample_gap_seconds"] for d in rows)})
    paired = []
    for steps, seeds in ((64, (42, 43, 44)), (256, (43, 44))):
        for seed in seeds:
            pair = {opt: next(d for d in results.values() if d["configuration"]["optimizer"] == opt
                              and d["configuration"]["lr"] == best[opt]
                              and d["configuration"]["seed"] == seed and d["configuration"]["steps"] == steps)
                    for opt in best}
            assert pair["adamw"]["training_window_indices"] == pair["gefen"]["training_window_indices"]
            assert pair["adamw"]["initial_tune"] == pair["gefen"]["initial_tune"]
            assert pair["adamw"]["initial_test"] == pair["gefen"]["initial_test"]
            paired.append({"steps": steps, "seed": seed,
                           "adamw_test_ce": pair["adamw"]["final_test"]["mean_ce"],
                           "gefen_test_ce": pair["gefen"]["final_test"]["mean_ce"],
                           "gefen_minus_adamw_test_ce": pair["gefen"]["final_test"]["mean_ce"] - pair["adamw"]["final_test"]["mean_ce"]})
    suite = (root / "minagi-gefen-paging-suite-final.log").read_text()
    assert "Ran 96 tests" in suite and "\nOK\n" in suite
    suite_power = json.loads((root / "suite-final.power.json").read_text())
    assert suite_power["returncode"] == 0 and not suite_power["errors"]
    environment = {"packages": {name: importlib.metadata.version(name) for name in
                                 ("torch", "triton", "gefen", "numba", "llvmlite", "quack-kernels", "sonic-moe")},
                   "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    (root / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    files = sorted(p for p in root.iterdir() if p.is_file() and p.suffix in (".json", ".log") and p.name != "SHA256SUMS.json")
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    manifest_path = root / "SHA256SUMS.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    archive = Path("replication/results/gefen_paging_evidence.tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        for path in [*files, manifest_path]:
            tar.add(path, arcname=path.name)
    with tarfile.open(archive) as tar:
        assert all(hashlib.sha256(tar.extractfile(name).read()).hexdigest() == digest for name, digest in manifest.items())
    report = {"tests_passed": 96, "selected_lr_by_tuning_seed_42": best, "comparison": comparison,
              "paired_test_ce": paired, "environment": environment,
              "runs": {name: {k: d[k] for k in ("configuration", "sources", "manifest_sha256", "text_sha256", "initial_tune", "final_tune", "initial_test", "final_test", "qualified_power", "power_source", "peak_allocated_gib")}
                       for name, d in results.items()},
              "evidence": {"files": len(files), "path": str(archive), "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()},
              "limits": ["Short BF16 adaptation, fixed depth 6; no native FP4 learning or pretraining convergence qualification.",
                         "Character partitions within TinyStories validation text; canonical split/document deduplication and original checkpoint data overlap are not established.",
                         "Seed 42 selects LR from six values at 64 steps; seeds 43/44 confirm. Longer runs reuse this LR, not a separate 256-step search.",
                         "Current AdamW uses slab-global counters; Gefen deliberately uses expert-local paused counters. This compares integrations, not isolated optimizer mathematics.",
                         "All touched experts remain cached in host RAM, with disk writes disabled; bounded-host-cache I/O requires separate performance measurement.",
                         "Reported training time/energy includes first calibration and new-expert partition inference but excludes initial/final held-out evaluation.",
                         "Selected-rate power uses an external monitor process, avoiding internal monitor GIL gaps during calibration. Raw internal records are also retained.",
                         "Results do not establish statistical equivalence or a consistent time-to-quality winner."]}
    Path("replication/results/gefen_paging.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"comparison": comparison, "paired": paired, "evidence": report["evidence"]}, indent=2))


if __name__ == "__main__":
    main()
