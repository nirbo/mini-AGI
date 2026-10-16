"""Profile training in memory, opening checkpoint files strictly read-only.

Unprofiled timings and a separate CUDA trace distinguish step throughput from
profiler overhead. Parameters/Adam state change only in RAM/VRAM, never on disk.
"""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from minagi.ingest import as_stream
from minagi.pool import PooledMLP
from minagi.precision import amp, set_compute_dtype
from minagi.recur import load_any


def union_duration(intervals):
    """Length of the union of (start, end) intervals, without double counting."""
    total, end = 0., float("-inf")
    for start, stop in sorted(intervals):
        total += max(0., stop - max(start, end))
        end = max(end, stop)
    return total


def trace_summary(path):
    events = json.loads(path.read_text())["traceEvents"]
    window = next(e for e in events if e.get("name") == "measured_training_step"
                  and e.get("ph") == "X")
    start, stop = window["ts"], window["ts"] + window["dur"]
    kernels, copies, names = [], [], Counter()
    for e in events:
        if e.get("ph") != "X" or "dur" not in e:
            continue
        a, b = max(start, e["ts"]), min(stop, e["ts"] + e["dur"])
        if b <= a:
            continue
        if e.get("cat") == "kernel":
            kernels.append((a, b))
            names[e["name"]] += b - a
        elif e.get("cat") in ("gpu_memcpy", "gpu_memset"):
            copies.append((a, b))
    if not kernels:
        raise RuntimeError("CUDA trace contains no kernels; GPU activity was not captured")
    return {"profiled_step_ms": window["dur"] / 1000,
            "kernel_count": len(kernels),
            "kernel_union_ms": union_duration(kernels) / 1000,
            "kernel_or_copy_union_ms": union_duration(kernels + copies) / 1000,
            "top_kernels_ms": [(name, dur / 1000) for name, dur in names.most_common(15)],
            "caution": "Instrumented timeline; profiler can increase host gaps. Union time is not SM occupancy or Tensor Core utilization."}


def training_state_digests(model, optimizer):
    """Exact A/B evidence; call outside timing, after an update and clipping."""
    hashes = {key: hashlib.sha256() for key in ("parameters", "clipped_gradients", "optimizer")}

    def add(group, name, tensor):
        h = hashes[group]
        h.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        h.update(tensor.detach().reshape(-1).contiguous().view(torch.uint8).cpu().numpy().tobytes())

    for name, param in model.named_parameters():
        add("parameters", name, param)
        if param.grad is not None:
            add("clipped_gradients", name, param.grad)
        for key, value in sorted(optimizer.state.get(param, {}).items()):
            add("optimizer", f"{name}/{key}", torch.as_tensor(value))
    return {**{key: h.hexdigest() for key, h in hashes.items()},
            "slot_expert_uids": [model.pool._f(e) if e >= 0 else -1 for e in model.pool.slots]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--text", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--stride", type=int, default=2048)
    ap.add_argument("--depth", type=int, default=24)
    ap.add_argument("--sample-depth", action="store_true")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    ap.add_argument("--dispatch", choices=("exact", "padded"), default="exact")
    ap.add_argument("--expert-backend", choices=("reference", "sonic"), default="reference")
    ap.add_argument("--no-checkpoint", action="store_true")
    ap.add_argument("--no-weight-cache", action="store_true",
                    help="cast expert weights on every use (reference path)")
    ap.add_argument("--no-profile", action="store_true")
    ap.add_argument("--deterministic", action="store_true",
                    help="require deterministic PyTorch algorithms, including cuBLAS")
    ap.add_argument("--state-digests", action="store_true",
                    help="hash parameters, clipped gradients and Adam state after every update (outside timing)")
    args = ap.parse_args()
    if min(args.context, args.stride, args.depth, args.steps, args.warmup) < 1:
        ap.error("context, stride, depth, steps and warmup must be positive")
    if not torch.cuda.is_available():
        ap.error("CUDA is required")
    os.environ["MINAGI_DISPATCH"] = args.dispatch
    os.environ["MINAGI_EXPERT_BACKEND"] = args.expert_backend
    if args.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    set_compute_dtype(args.precision)
    model, _ = load_any(str(args.weights), torch.device("cuda"), read_only=True)
    model.pool.cache_training_weights = not args.no_weight_cache
    if args.context > model.cfg.block or args.depth > model.cfg.max_steps:
        ap.error("context/depth may not exceed the checkpoint ceilings")
    model.cfg.max_steps = args.depth
    model.cfg.bptt_window = args.depth
    if not args.sample_depth:
        model.cfg.train_steps_mean = 0
    for module in model.modules():
        if isinstance(module, PooledMLP):
            module.grad_checkpoint = not args.no_checkpoint
    model.train()
    # Match read's optimizer family, clipping and objective terms. This is a
    # fresh optimizer, not a numerically faithful resumed learning experiment.
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, betas=(0.9, 0.95),
                            weight_decay=0.1, fused=True)
    model.pool.attach_optimiser(opt)
    needed = args.context + args.stride * (args.steps + args.warmup + 1) + 1
    data = as_stream(str(args.text))[:needed]
    if len(data) < needed:
        ap.error(f"text needs at least {needed} encoded tokens for this run")
    ids = torch.tensor(data.astype(np.int64), device="cuda").unsqueeze(0)
    depths = []
    original_depth = model.sample_depth

    def sample_depth():
        depth = original_depth()
        depths.append(depth)
        return depth

    model.sample_depth = sample_depth

    def step(i):
        start = i * args.stride
        opt.zero_grad(set_to_none=True)
        with amp("cuda"):
            _, loss = model(ids[:, start:start + args.context],
                            ids[:, start + 1:start + args.context + 1])
            loss = loss + model.cfg.pool_aux * model.pool_aux() + model.pool_balance()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()
        return loss.detach(), norm.detach()

    times, losses, norms, routed, expected_depths, digests = [], [], [], [], [], []
    torch.cuda.reset_peak_memory_stats()
    for i in range(args.warmup + args.steps):
        if i == args.warmup:
            measurement_start = time.time()
        torch.cuda.synchronize()
        start = time.perf_counter()
        loss, norm = step(i)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        _, _, assignments = model.pool_dropped()
        if not torch.isfinite(loss) or not torch.isfinite(norm):
            raise RuntimeError("non-finite training loss or gradient norm")
        if i >= args.warmup:
            times.append(elapsed)
            losses.append(float(loss))
            norms.append(float(norm))
            routed.append(assignments)
            expected_depths.append(model.last_steps)
        if args.state_digests:
            digests.append({"step": i, **training_state_digests(model, opt)})
        print(f"step {i}: {elapsed:.4f}s, depth {depths[-1]}, loss {float(loss):.4f}", flush=True)
    measurement_end = time.time()
    peak = torch.cuda.max_memory_allocated() / 2**30
    result = {"configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "gpu": torch.cuda.get_device_name(), "compute_capability": torch.cuda.get_device_capability(),
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "precision": args.precision, "master_weight_dtype": str(model.pool.w1.dtype),
              "experts": model.pool.n_experts(), "resident": model.pool.resident,
              "halting_freeze": model.cfg.halt_freeze,
              "selection_temperature": model.pool.select_temperature,
              "balance_weight": model.pool.balance,
              "step_seconds": times, "median_step_seconds": statistics.median(times),
              "timing_interval_unix": [measurement_start, measurement_end],
              "processed_tokens_per_second": args.context / statistics.median(times),
              "new_tokens_per_second": args.stride / statistics.median(times),
              "sampled_depths": depths[args.warmup:], "losses": losses, "gradient_norms": norms,
              "training_state_digests": digests,
              "routed_assignments_per_step": routed, "expected_depths": expected_depths,
              "tf32_matmul_allowed": torch.backends.cuda.matmul.allow_tf32,
              "peak_allocated_gib": peak,
              "manifest_sha256": hashlib.sha256((args.weights / "manifest.json").read_bytes()).hexdigest(),
              "input_ids_sha256": hashlib.sha256(data.astype("<u2").tobytes()).hexdigest(),
              "limits": ["Warm RAM/filesystem cache; excludes checkpoint loading, saves, evaluation and sampling.",
                         "Trained checkpoint, real text, fresh optimizer, few steps: no convergence/quality claim.",
                         "Full windows overlap by context minus stride, so processed and new tokens/s differ.",
                         "FLOP counts cover profiler-supported operators; they are not complete model FLOPs or MFU."]}
    if args.state_digests:
        result["limits"].append("Per-step state hashing transfers data outside timing; use separate runs for performance comparisons.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Save timing evidence even if profiling is unavailable on this machine.
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    if not args.no_profile:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=True, with_flops=True) as prof:
            with torch.profiler.record_function("measured_training_step"):
                step(args.warmup + args.steps)
                torch.cuda.synchronize()
        trace = args.out.with_suffix(".trace.json")
        prof.export_chrome_trace(str(trace))
        result["profile"] = trace_summary(trace)
        events = prof.key_averages()
        result["profile"]["operators"] = [
            {"name": e.key, "calls": e.count, "self_cpu_ms": e.self_cpu_time_total / 1000,
             "self_cuda_ms": e.self_device_time_total / 1000, "flops": e.flops}
            for e in sorted((e for e in events if e.key.startswith("aten::")),
                            key=lambda e: e.self_device_time_total, reverse=True)[:25]]
        result["profile"]["cpu_events"] = [
            {"name": e.key, "calls": e.count, "self_cpu_ms": e.self_cpu_time_total / 1000}
            for e in sorted(events, key=lambda e: e.self_cpu_time_total, reverse=True)[:15]]
        shapes, rows = Counter(), Counter()
        for e in prof.events():
            if e.name in ("aten::mm", "aten::bmm"):
                shapes[(e.name, str(e.input_shapes))] += 1
            if e.name == "aten::mm" and len(e.input_shapes) == 2:
                a, b = e.input_shapes
                if len(a) == len(b) == 2 and a[1] == b[0] and (a[1], b[1]) in (
                        (model.cfg.d_model, model.cfg.pool_d_ff),
                        (model.cfg.pool_d_ff, model.cfg.d_model)):
                    rows[a[0]] += 1
        result["profile"]["common_matmul_shapes"] = [
            {"op": key[0], "shapes": key[1], "calls": n}
            for key, n in shapes.most_common(20)]
        result["profile"]["expert_shaped_matmul_rows"] = dict(sorted(rows.items()))
        if rows:
            result["profile"]["round_rows_to_128_work_multiplier"] = (
                sum(((m + 127) // 128) * 128 * n for m, n in rows.items())
                / sum(m * n for m, n in rows.items()))
        result["profile"]["trace_path"] = str(trace)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    if model.pool.tiers.writebacks or model.pool.tiers.dirty:
        raise RuntimeError("profiling must not write checkpoint files")
    print(json.dumps({k: v for k, v in result.items() if k != "profile"}, indent=2))


if __name__ == "__main__":
    main()
