"""CPU-only summary/archive of the complete-layer NVFP4 crossover sweep."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


def main():
    root = Path("runs/performance/nvfp4-crossover")
    cases = defaultdict(list)
    files = sorted(p for p in root.glob("d*.json") if not p.name.endswith(".power.json"))
    for path in files:
        r = json.loads(path.read_text())
        cfg = r["configuration"]
        key = tuple(cfg[k] for k in ("hidden", "width", "experts", "rows_per_expert"))
        cases[key].append(r)
        log = Path("/tmp/minagi-cross-" + path.stem + ".log")
        if log.exists():
            shutil.copyfile(log, path.with_suffix(".log"))
    rows = []
    for shape, results in sorted(cases.items()):
        row = {"hidden_width_experts_rows_per_expert": list(shape), "backends": {}}
        for backend in ("sonic", "nvfp4"):
            selected = [r for r in results if r["configuration"]["backend"] == backend]
            good = [r["measurement"] for r in selected if r["status"] == "passed"]
            out = {"statuses": [r["status"] for r in selected]}
            if good:
                duration = sum(m["end_monotonic"] - m["start_monotonic"] for m in good)
                tokens = sum(m["token_rows"] for m in good)
                energy = sum(m["power"]["estimated_board_energy_j"] for m in good)
                out.update(aggregate_input_rows_per_second=tokens / duration,
                           aggregate_board_joules_per_input_row=energy / tokens,
                           average_board_power_w=energy / duration,
                           peak_sampled_board_power_w=max(m["power"]["peak_sampled_board_power_w"] for m in good),
                           median_step_ms_range=[min(m["median_seconds"] for m in good) * 1000,
                                                 max(m["median_seconds"] for m in good) * 1000],
                           peak_allocated_gib=max(m["peak_allocated_gib"] for m in good),
                           optimizer_state_gib=selected[0].get("optimizer_state_gib"))
            row["backends"][backend] = out
        s, n = row["backends"]["sonic"], row["backends"]["nvfp4"]
        if "aggregate_input_rows_per_second" in s and "aggregate_input_rows_per_second" in n:
            row.update(nvfp4_throughput_ratio_to_sonic=n["aggregate_input_rows_per_second"] / s["aggregate_input_rows_per_second"],
                       nvfp4_energy_ratio_to_sonic=n["aggregate_board_joules_per_input_row"] / s["aggregate_board_joules_per_input_row"])
        rows.append(row)
    report = {"cases": rows, "raw_runs": [r for result in cases.values() for r in result],
              "limits": ["Current unfused native FP4 preparation versus qualified Sonic, not a universal hardware crossover.",
                         "Complete expert-layer forward/backward plus optimizer/cache refresh; other model components are excluded.",
                         "24 GiB process allocation budget reserves desktop and additional-model headroom; allocation-cap failures are not physical 32 GiB capacity claims.",
                         "Full FP32 Adam state is allocated; all NVFP4 projection shapes pass independent decoded-operand checks before timing.",
                         "Two ABBA observations per backend, 3-second measurement windows after warmup/settling. Numerical learning quality is not measured."]}
    retained = sorted(p for p in root.iterdir() if p.is_file() and p.suffix in (".json", ".log") and p.name != "SHA256SUMS.json")
    checksums = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in retained}
    manifest = root / "SHA256SUMS.json"
    manifest.write_text(json.dumps(checksums, indent=2) + "\n")
    archive = Path("replication/results/nvfp4_crossover_evidence.tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        for path in [*retained, manifest]:
            tar.add(path, arcname=path.name)
    with tarfile.open(archive) as tar:
        assert all(hashlib.sha256(tar.extractfile(name).read()).hexdigest() == digest for name, digest in checksums.items())
    report["archive"] = {"path": str(archive), "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "files": len(retained)}
    Path("replication/results/nvfp4_crossover.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
