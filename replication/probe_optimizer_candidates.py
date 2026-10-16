"""Isolated optimizer state/latency and BF16 update probes; no model files.

Use the main repository venv. GPU work, including Gefen JIT, is monitored.
This is not integration with mini-AGI's Adam-specific paging format.
"""

import argparse
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time
import traceback

import torch

from replication.gpu_monitor import PowerMonitor, summarize_samples


def state_bytes(optimizer):
    seen, total = set(), 0
    def visit(value):
        nonlocal total
        if torch.is_tensor(value):
            storage = value.untyped_storage()
            key = (value.device, storage.data_ptr())
            if key not in seen:
                seen.add(key)
                total += storage.nbytes()
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
    visit(optimizer.state_dict())
    return total


def make_optimizer(name, weights, lr=1e-5):
    if name == "adamw":
        return torch.optim.AdamW(weights, lr=lr, betas=(.9, .95), weight_decay=0., fused=True)
    if name == "muon":
        return torch.optim.Muon(weights, lr=lr, momentum=.95, weight_decay=0., adjust_lr_fn="match_rms_adamw")
    if name == "gefen":
        from gefen import Gefen
        return Gefen(weights, lr=lr, betas=(.9, .95), weight_decay=0., fused=True)
    if name == "gefen_muon":
        from gefen import GefenMuon
        return GefenMuon(weights, lr=lr, momentum=.95, weight_decay=0., fused=True,
                         adjust_lr_fn="match_rms_adamw")
    raise ValueError(name)


def latency(args):
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    shapes = [(args.width, args.hidden), (args.width, args.hidden), (args.hidden, args.width)] * args.experts
    weights = [torch.nn.Parameter(torch.empty(s, device="cuda", dtype=dtype).normal_(std=s[-1]**-.5)) for s in shapes]
    # Correlated row/column magnitudes exercise Gefen partition inference.
    for p in weights:
        row = torch.randn(p.shape[0], 1, device="cuda", dtype=dtype)
        col = torch.randn(1, p.shape[1], device="cuda", dtype=dtype)
        p.grad = (row * col + .1 * torch.randn_like(p)) * 1e-3
    opt = make_optimizer(args.optimizer, weights)
    t0 = time.perf_counter()
    opt.step()
    torch.cuda.synchronize()
    startup = time.perf_counter() - t0
    # Check loss of small parameter updates using a snapshot outside timing.
    before = [p.detach().clone() for p in weights]
    opt.step()
    torch.cuda.synchronize()
    moved = sum(int((p != old).count_nonzero()) for p, old in zip(weights, before))
    count = sum(p.numel() for p in weights)
    del before
    for _ in range(5):
        opt.step()
    torch.cuda.synchronize()
    settle = time.perf_counter()
    while time.perf_counter() - settle < 1.1:
        opt.step()
        torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times, start = [], time.perf_counter()
    while time.perf_counter() - start < args.seconds or len(times) < 3:
        t = time.perf_counter()
        opt.step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    end = time.perf_counter()
    if not all(bool(torch.isfinite(p).all()) for p in weights):
        raise AssertionError("non-finite optimized parameter")
    return {"parameter_count": count, "optimizer_state_bytes": state_bytes(opt),
            "state_bytes_per_parameter": state_bytes(opt) / count,
            "moved_fraction_on_second_update": moved / count, "first_step_including_jit_seconds": startup,
            "median_step_seconds": sorted(times)[len(times) // 2],
            "iterations": len(times), "start_monotonic": start, "end_monotonic": end,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "state_keys": sorted({k for state in opt.state.values() for k in state}),
            "automatic_periods": [int(s["automatic_period"]) for s in opt.state.values() if "automatic_period" in s]}


def update_retention(args):
    """Same positive adaptive direction at tiny LR; not a learning comparison."""
    cases = []
    for dtype in (torch.float32, torch.bfloat16):
        torch.manual_seed(42)
        p = torch.nn.Parameter(torch.full((128, 128), .013, device="cuda", dtype=dtype))
        initial = p.detach().float().clone()
        opt = make_optimizer(args.optimizer, [p], lr=1.8e-5)
        for _ in range(32):
            p.grad = torch.ones_like(p)
            opt.step()
        # Ordinary optimizer checkpoint continuation. This does not test the
        # application's separate per-expert paging/serialization mechanism.
        q = torch.nn.Parameter(p.detach().clone())
        resumed = make_optimizer(args.optimizer, [q], lr=1.8e-5)
        resumed.load_state_dict(copy.deepcopy(opt.state_dict()))
        p.grad, q.grad = torch.ones_like(p), torch.ones_like(q)
        opt.step()
        resumed.step()
        torch.testing.assert_close(p, q, rtol=0, atol=0)
        cases.append({"dtype": str(dtype), "changed_fraction": float((p.float() != initial).float().mean()),
                      "mean_delta": float((p.float() - initial).mean()),
                      "state_bytes_per_parameter": state_bytes(opt) / p.numel(),
                      "standard_checkpoint_next_update_bit_identical": True})
    return {"cases": cases, "limits": "Artificial constant-gradient update retention; optimizer directions/scaling differ. Not matched learning quality."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("latency", "retention"), required=True)
    parser.add_argument("--optimizer", choices=("adamw", "gefen", "muon", "gefen_muon"), required=True)
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=3.)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = {"configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "torch": torch.__version__, "gefen": importlib.metadata.version("gefen"),
              "gefen_commit": "e735f851a7ebdf0106da87126458c86e3e583d99",
              "limits": ["Optimizer-only synthetic fixed gradients; matrix list preserves independent experts.",
                         "No model forward/backward, paging or convergence measurement. Optimizer latency is not end-to-end throughput.",
                         "Different optimizer mathematics cannot be compared as a correctness-equivalent replacement."]}
    with PowerMonitor() as monitor:
        try:
            torch.manual_seed(42)
            torch.set_num_threads(4)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.cuda.set_per_process_memory_fraction(.75)
            report["result"] = {"latency": latency, "retention": update_retention}[args.mode](args)
            report["status"] = "passed"
        except BaseException as exc:
            report.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
            raise
        finally:
            monitor.sample()
            result = report.get("result", {})
            if "start_monotonic" in result:
                result["power"] = summarize_samples(monitor.samples, result["start_monotonic"], result["end_monotonic"])
                result["estimated_board_joules_per_optimizer_step"] = result["power"]["estimated_board_energy_j"] / result["iterations"]
            monitor.save(args.out.with_suffix(".power.json"))
            args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report.get("result", {})), flush=True)


if __name__ == "__main__":
    main()
