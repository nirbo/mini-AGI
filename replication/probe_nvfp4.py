"""Standalone NVFP4 kernel/training probes; never loads or changes model files.

Requires Quartet-II plus FlashInfer in the repository venv. All GPU work is
power monitored, including warmup/compilation. Benchmark windows exclude those
costs and include quantization when testing a complete linear training pass.
"""

import argparse
from contextlib import contextmanager
import importlib.metadata
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

import torch
import torch.nn.functional as F

from replication.gpu_monitor import PowerMonitor, summarize_samples


def provenance():
    """Identify the executed probe, including edits not represented by HEAD."""
    paths = [Path(__file__), Path(__file__).with_name("probe_moe_backends.py"),
             Path(__file__).with_name("gpu_monitor.py")]
    return {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
            "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            "environment": {k: os.environ.get(k) for k in
                            ("FLASHINFER_EXTRA_CUDAFLAGS", "MAX_JOBS", "CUDA_VISIBLE_DEVICES")}}


def error_metrics(actual, reference):
    a, r = actual.detach().float().flatten(), reference.detach().float().flatten()
    if not bool(torch.isfinite(a).all() & torch.isfinite(r).all()):
        return {"finite": False, "relative_l2": None, "cosine": None, "max_abs": None}
    denominator = r.norm().clamp_min(1e-30)
    return {"finite": bool(torch.isfinite(a).all()),
            "relative_l2": float((a - r).norm() / denominator),
            "cosine": float(F.cosine_similarity(a, r, dim=0)),
            "max_abs": float((a - r).abs().max())}


def dequantize(packed, scales, tensor_scale):
    """Independent E2M1/128x4-scale decoder, evaluated in FP32."""
    codes = torch.stack((packed & 15, packed >> 4), -1).flatten(-2).long()
    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6],
                       device=packed.device, dtype=torch.float32)
    rows, cols = codes.shape
    # Physical 128x4 layout: [row_block, col_block, row%32,
    # (row%128)//32, col%4]. Decode indices directly, independently of Quartet.
    r = torch.arange(rows, device=packed.device)[:, None]
    c = torch.arange(cols // 16, device=packed.device)[None, :]
    index = ((r // 128 * (cols // 64) + c // 4) * 512
             + (r % 32) * 16 + ((r % 128) // 32) * 4 + c % 4)
    sf = scales.flatten().float()[index].repeat_interleave(16, -1)
    return lut[codes] * sf * tensor_scale


def raw_mm(xq, wq, xs, ws, alpha, backend):
    from flashinfer import mm_fp4
    k = xq.shape[1] * 2
    return mm_fp4(xq, wq.T, xs.reshape(-1, k // 16), ws.reshape(-1, k // 16).T,
                  alpha, out_dtype=torch.bfloat16, backend=backend, block_size=16,
                  use_8x4_sf_layout=False, use_nvfp4=True)


def benchmark(fn, rows, seconds, warmup=3, settle_seconds=1.1):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    # Board power may average the previous second. Let the same workload run
    # long enough to replace the preceding phase in that sensor window.
    settled = time.perf_counter()
    while time.perf_counter() - settled < settle_seconds:
        fn()
        torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times, start = [], time.perf_counter()
    while len(times) < 3 or time.perf_counter() - start < seconds:
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    end = time.perf_counter()
    return {"start_monotonic": start, "end_monotonic": end, "iterations": len(times),
            "power_settle_seconds": settle_seconds,
            "median_seconds": statistics.median(times), "min_seconds": min(times),
            "p90_seconds": sorted(times)[int(.9 * (len(times) - 1))],
            "token_rows_per_second": rows * len(times) / (end - start),
            "token_rows": rows * len(times),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}


def check_kernel(metrics):
    # Quantization error is excluded: compare to the SAME decoded FP4 operands.
    if not metrics["finite"] or metrics["relative_l2"] > .005:
        raise AssertionError(f"FP4 GEMM differs from decoded-operand reference: {metrics}")


def print_case(case):
    """Keep compiler logs in JSON artifacts rather than flooding the terminal."""
    brief = {k: v for k, v in case.items() if k != "traceback"}
    if "error" in brief:
        brief["error"] = brief["error"][:800]
    print(json.dumps(brief), flush=True)


def kernel_probes(report, args):
    from quartet2.quant import quant_fp4, NVFP4QuantMode
    for backend in args.backends:
        for m, k, n in ((128, 512, 2048), (128, 2048, 512)):
            case = {"kind": "lossless_operands", "backend": backend, "shape_mkn": [m, k, n]}
            report["cases"].append(case)
            try:
                xq = torch.full((m, k // 2), 0x22, device="cuda", dtype=torch.uint8)
                wq = torch.full((n, k // 2), 0x22, device="cuda", dtype=torch.uint8)
                xs = torch.ones(m, k // 16, device="cuda", dtype=torch.float8_e4m3fn)
                ws = torch.ones(n, k // 16, device="cuda", dtype=torch.float8_e4m3fn)
                alpha = torch.ones((), device="cuda", dtype=torch.float32)
                y = raw_mm(xq, wq, xs, ws, alpha, backend)
                torch.cuda.synchronize()
                torch.testing.assert_close(y, torch.full_like(y, k), rtol=0, atol=0)
                case["status"] = "passed"
            except Exception as exc:
                case.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
                print_case(case)
                break
            print_case(case)
        if case["status"] == "failed":
            continue
        for rows in args.rows:
            for k, n in ((512, 2048), (2048, 512)):
                m = ((rows + 127) // 128) * 128
                case = {"kind": "random_operands", "backend": backend,
                        "logical_rows": rows, "padded_shape_mkn": [m, k, n]}
                report["cases"].append(case)
                try:
                    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
                    x[rows:] = 0
                    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / k**.5
                    xq = quant_fp4(x, scale_override=1., mode=NVFP4QuantMode.FOUR_SIX)
                    wq = quant_fp4(w, scale_override=1., mode=NVFP4QuantMode.FOUR_SIX)
                    alpha = xq.tensor_scale * wq.tensor_scale
                    ref = dequantize(*xq) @ dequantize(*wq).T
                    fn = lambda: raw_mm(xq.fp4, wq.fp4, xq.micro_scales, wq.micro_scales, alpha, backend)
                    y = fn()
                    case["kernel_error"] = error_metrics(y, ref)
                    check_kernel(case["kernel_error"])
                    case["including_quantization_error"] = error_metrics(y[:rows], x[:rows].float() @ w.float().T)
                    if rows in (128, 1024):
                        case["packed_gemm_benchmark"] = benchmark(fn, rows, args.seconds)
                    case["status"] = "passed"
                except Exception as exc:
                    case.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
                print_case(case)


@contextmanager
def capture_fp4_mm():
    """Record actual forward/dgrad/wgrad operands without altering arithmetic."""
    import quartet2.linear as linear
    original, records = linear._fp4_mm, []

    def record(xq, wq, xs, ws, alpha):
        y = original(xq, wq, xs, ws, alpha)
        records.append((xq, wq, xs, ws, alpha, y.detach()))
        return y

    linear._fp4_mm = record
    try:
        yield records
    finally:
        linear._fp4_mm = original


def training_probes(report, args):
    from quartet2.linear import Quartet_II_fn, get_hadamard_matrix
    from quartet2.quant import NVFP4QuantMode
    had = get_hadamard_matrix(128, torch.bfloat16, torch.device("cuda"))
    shapes = [tuple(map(int, shape.split(","))) for shape in args.shapes]
    for m, k, n in shapes:
        case = {"kind": "quartet_forward_backward", "shape_mkn": [m, k, n]}
        report["cases"].append(case)
        x = torch.randn(m, k, device="cuda", requires_grad=True)
        w = (torch.randn(n, k, device="cuda") / k**.5).requires_grad_()
        g = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)

        def forward():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return Quartet_II_fn.apply(x, w, had, NVFP4QuantMode.FOUR_SIX, False)

        def step():
            x.grad = w.grad = None
            forward().backward(g)

        def reference_step():
            x.grad = w.grad = None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                (x.to(torch.bfloat16) @ w.to(torch.bfloat16).T).backward(g)

        try:
            with capture_fp4_mm() as records:
                step()
            torch.cuda.synchronize()
            if len(records) != 3:
                raise AssertionError(f"Expected forward, input-gradient, weight-gradient FP4 GEMMs, got {len(records)}")
            case["three_native_gemm_errors"] = []
            for xq, wq, xs, ws, alpha, y in records:
                ref = (dequantize(xq, xs, torch.ones_like(alpha)) @
                       dequantize(wq, ws, torch.ones_like(alpha)).T) * alpha
                error = error_metrics(y, ref)
                check_kernel(error)
                case["three_native_gemm_errors"].append(error)
            case["master_and_gradient_dtypes"] = [str(t.dtype) for t in (x, w, x.grad, w.grad)]
            if any(t.dtype != torch.float32 or not bool(torch.isfinite(t).all()) for t in (x, w, x.grad, w.grad)):
                raise AssertionError("FP32 masters/gradients must remain finite")
            # Recover each operand's scale separately for the gradient oracle.
            from quartet2.quant import quant_fp4
            qx = quant_fp4(x.detach().to(torch.bfloat16), scale_override=1., mode=NVFP4QuantMode.FOUR_SIX)
            qw = quant_fp4(w.detach().to(torch.bfloat16), scale_override=1., mode=NVFP4QuantMode.FOUR_SIX)
            decoded_x, decoded_w = dequantize(*qx), dequantize(*qw)
            dx_quant_ref = g.float() @ decoded_w
            dw_quant_ref = g.float().T @ decoded_x
            reference_step()
            dx_ref, dw_ref = x.grad.clone(), w.grad.clone()
            dx, dw = torch.zeros_like(x), torch.zeros_like(w)
            for _ in range(args.gradient_samples):
                step()
                dx += x.grad
                dw += w.grad
            case["mean_gradient_error_vs_bf16"] = {
                "samples": args.gradient_samples,
                "input": error_metrics(dx / args.gradient_samples, dx_ref),
                "weight": error_metrics(dw / args.gradient_samples, dw_ref)}
            case["mean_gradient_error_vs_decoded_forward"] = {
                "input": error_metrics(dx / args.gradient_samples, dx_quant_ref),
                "weight": error_metrics(dw / args.gradient_samples, dw_quant_ref)}
            case["bf16_complete_training_benchmark"] = benchmark(reference_step, m, args.seconds)
            case["nvfp4_complete_training_benchmark"] = benchmark(step, m, args.seconds)
            case["status"] = "passed"
        except Exception as exc:
            case.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
        print_case(case)


def grouped_probes(report, args):
    from flashinfer.grouped_mm import grouped_mm_fp4
    from flashinfer.gemm import group_gemm_nvfp4_nt_groupwise
    from quartet2.quant import quant_fp4, NVFP4QuantMode
    for counts in ([128, 0, 256, 128], [1, 31, 127, 225],
                   [0, 1, 17, 31, 63, 127, 129, 257] + [1339] * 23 + [1346]):
        # The SM120 CUTLASS wrapper requires offsets divisible by four. Scale
        # tiles use a separate segmented layout, not one swizzle over all rows.
        e, k, n = len(counts), 512, 2048
        x = torch.randn(sum(counts), k, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(e * n, k, device="cuda", dtype=torch.bfloat16) / k**.5
        amax = x.abs().max().float()
        qw = quant_fp4(w, scale_override=1., mode=NVFP4QuantMode.FOUR_SIX)
        dw = dequantize(*qw).reshape(e, n, k)
        for backend in args.backends:
            # cuDNN's published tests use globally swizzled scales and 128-row
            # boundaries. Qualify that supported subset with explicit padding.
            # CUTLASS supports four-row boundaries with segmented SF storage.
            alignment = 128 if backend == "cudnn" else 4
            padded = [(c + alignment - 1) // alignment * alignment for c in counts]
            starts = [0]
            for c in padded:
                starts.append(starts[-1] + c)
            m = sum(padded)
            sf_starts = (starts[:-1] if backend == "cudnn" else
                         [(s + i * 127) // 128 * 128 for i, s in enumerate(starts[:-1])])
            sf_rows = max(s + (c + 127) // 128 * 128 for s, c in zip(sf_starts, padded))
            sf = torch.zeros(sf_rows, k // 16, device="cuda", dtype=torch.float8_e4m3fn)
            chunks, refs, logical_start = [], [], 0
            for expert, count in enumerate(padded):
                if not count:
                    continue
                xp = torch.zeros((count + 127) // 128 * 128, k, device="cuda", dtype=torch.bfloat16)
                logical_count = counts[expert]
                xp[:logical_count] = x[logical_start:logical_start + logical_count]
                logical_start += logical_count
                qx = quant_fp4(xp, scale_override=1., mode=NVFP4QuantMode.FOUR_SIX, amax=amax)
                chunks.append(qx.fp4[:count])
                sf[sf_starts[expert]:sf_starts[expert] + xp.shape[0]].copy_(qx.micro_scales)
                refs.append(dequantize(*qx)[:count] @ dw[expert].T)
            packed, ref = torch.cat(chunks), torch.cat(refs)
            offsets = torch.tensor(starts, device="cuda", dtype=torch.int32)
            alpha = (qx.tensor_scale * qw.tensor_scale).reshape(1)
            case = {"kind": "grouped_nvfp4_forward", "backend": backend, "counts": counts,
                    "padded_counts": padded, "scale_storage_rows": sf_rows,
                    "shape_total_mekn": [m, e, k, n]}
            report["cases"].append(case)
            try:
                if backend == "cudnn":
                    fn = lambda: grouped_mm_fp4(packed, qw.fp4.reshape(e, n, k // 2),
                                               sf, qw.micro_scales.reshape(e, n, k // 16),
                                               offsets, alpha=alpha, backend="cudnn")
                else:
                    alphas = alpha.expand(e).contiguous()
                    fn = lambda: group_gemm_nvfp4_nt_groupwise(
                        packed, qw.fp4.reshape(e, n, k // 2), sf.view(torch.uint8),
                        qw.micro_scales.reshape(e, n, k // 16).view(torch.uint8),
                        offsets, alpha=alphas)
                y = fn()
                case["kernel_error"] = error_metrics(y, ref)
                check_kernel(case["kernel_error"])
                case["packed_grouped_forward_benchmark"] = benchmark(fn, sum(counts), args.seconds)
                case["status"] = "passed"
            except Exception as exc:
                case.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
            print_case(case)


def causal_scale_probe(report, args):
    from quartet2.linear import Quartet_II_fn, get_hadamard_matrix
    from quartet2.quant import quant_fp4, NVFP4QuantMode
    prefix = torch.randn(128, 512, device="cuda", dtype=torch.bfloat16)
    full = torch.cat((prefix, torch.full_like(prefix, 100)))
    weight = torch.randn(2048, 512, device="cuda", dtype=torch.bfloat16) / 512**.5
    had = get_hadamard_matrix(128, torch.bfloat16, torch.device("cuda"))
    fixed = torch.tensor(128., device="cuda")
    case = {"kind": "forward_causality_scale_diagnostic"}
    report["cases"].append(case)
    for label, amax in (("dynamic_whole_matrix", None), ("fixed_calibrated_bound", fixed)):
        decoded, outputs = [], []
        for x in (prefix, full):
            q = quant_fp4(x, scale_override=1., mode=NVFP4QuantMode.FOUR_SIX, amax=amax)
            decoded.append(dequantize(*q)[:128])
            with torch.no_grad():
                outputs.append(Quartet_II_fn.apply(x, weight, had, NVFP4QuantMode.FOUR_SIX,
                                                 False, None, amax)[:128])
        case[label] = {"decoded_prefix_identical": torch.equal(*decoded),
                       "output_difference": error_metrics(outputs[1], outputs[0])}
    if not case["fixed_calibrated_bound"]["decoded_prefix_identical"]:
        raise AssertionError("Fixed input scale must preserve prefix quantization")
    if case["fixed_calibrated_bound"]["output_difference"]["max_abs"] != 0:
        raise AssertionError("Fixed input scale must preserve prefix output")
    case["status"] = "passed"
    print_case(case)


def learning_probe(report, args):
    """Matched synthetic regression: a feasibility check, not LLM quality."""
    from quartet2.linear import Quartet_II_fn, get_hadamard_matrix
    from quartet2.quant import NVFP4QuantMode
    h, width, batch, train_rows = 512, 2048, 128, 3072
    inputs = torch.randn(4096, h, device="cuda")
    teacher = torch.randn(width, h, device="cuda") / h**.5
    targets = inputs @ teacher.T
    initial = torch.randn_like(teacher) / h**.5
    generator = torch.Generator().manual_seed(314)
    batches = [torch.randint(train_rows, (batch,), generator=generator).to("cuda")
               for _ in range(args.learn_steps)]
    had = get_hadamard_matrix(128, torch.bfloat16, torch.device("cuda"))
    bound = torch.tensor(8., device="cuda")
    for backend in ("bf16", "quartet_nvfp4"):
        weight = torch.nn.Parameter(initial.clone())
        opt = torch.optim.AdamW([weight], lr=.003, weight_decay=0., fused=True)

        def predict(x):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if backend == "bf16":
                    return x.to(torch.bfloat16) @ weight.to(torch.bfloat16).T
                return Quartet_II_fn.apply(x, weight, had, NVFP4QuantMode.FOUR_SIX, False, None, bound)

        curve = []
        for i in range(args.learn_steps + 1):
            if i % 50 == 0 or i == args.learn_steps:
                with torch.no_grad():
                    actual = predict(inputs[train_rows:]).float()
                    deployed_mse = float(F.mse_loss(actual, targets[train_rows:]))
                    master_mse = float(F.mse_loss(inputs[train_rows:] @ weight.T, targets[train_rows:]))
                curve.append({"step": i, "heldout_compute_mse": deployed_mse, "heldout_fp32_master_mse": master_mse})
            if i == args.learn_steps:
                break
            ids = batches[i]
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(predict(inputs[ids]).float(), targets[ids])
            loss.backward()
            if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(weight.grad).all()):
                raise AssertionError("Non-finite synthetic training loss/gradient")
            opt.step()
        result = {"kind": "matched_synthetic_learning", "backend": backend, "steps": args.learn_steps,
                  "shape": [batch, h, width], "curve": curve, "activation_bound": 8.,
                  "master_dtype": str(weight.dtype), "status": "passed"}
        if curve[-1]["heldout_compute_mse"] >= curve[0]["heldout_compute_mse"] * .25:
            raise AssertionError("Synthetic heldout error did not fall by at least 75%")
        # Quantize BF16-trained weights on the same inference path for a small
        # apples-to-apples PTQ control (no claim beyond this regression task).
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                qout = Quartet_II_fn.apply(inputs[train_rows:], weight, had, NVFP4QuantMode.FOUR_SIX, False, None, bound)
            result["heldout_nvfp4_mse"] = float(F.mse_loss(qout.float(), targets[train_rows:]))
        report["cases"].append(result)
        print_case(result)


def attach_power(value, samples):
    if isinstance(value, dict):
        if "start_monotonic" in value and "iterations" in value:
            value["power"] = summarize_samples(samples, value["start_monotonic"], value["end_monotonic"])
            value["estimated_board_joules_per_token_row"] = value["power"]["estimated_board_energy_j"] / value["token_rows"]
        else:
            for child in value.values():
                attach_power(child, samples)
    elif isinstance(value, list):
        for child in value:
            attach_power(child, samples)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", choices=("kernel", "training", "grouped", "causality", "learning"), required=True)
    ap.add_argument("--backends", nargs="+", choices=("cudnn", "cutlass", "cute-dsl"))
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 31, 127, 128, 129, 1024])
    ap.add_argument("--seconds", type=float, default=2.)
    ap.add_argument("--gradient-samples", type=int, default=32)
    ap.add_argument("--shapes", nargs="+", default=["128,512,2048", "1024,512,2048", "128,2048,512", "1024,2048,512"])
    ap.add_argument("--learn-steps", type=int, default=400)
    args = ap.parse_args()
    args.backends = args.backends or (["cudnn", "cutlass"] if args.mode == "grouped" else
                                      ["cudnn", "cutlass", "cute-dsl"])
    if args.mode == "grouped" and "cute-dsl" in args.backends:
        ap.error("the grouped probe supports cudnn and cutlass only")
    # JIT backends invoke tools such as ninja installed in this same venv.
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    if args.seconds <= 0 or args.gradient_samples <= 0 or min(args.rows) <= 0 or args.learn_steps <= 0:
        ap.error("seconds, rows and gradient samples must be positive")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = {"configuration": vars(args) | {"out": str(args.out)}, "cases": [],
              **provenance(),
              "packages": {p: importlib.metadata.version(p) for p in
                           ("torch", "triton", "quartet2", "flashinfer-python", "nvidia-cudnn-frontend", "nvidia-cutlass-dsl")},
              "limits": ["Standalone synthetic linear operations, not model training throughput or convergence.",
                         "All masters and accumulated gradients FP32; matrix operands FP4 for Quartet forward AND backward.",
                         "Kernel comparisons use decoded FP4 operands; quantization error is reported separately.",
                         "Quartet currently pads token rows to multiples of 128; timing includes that arithmetic.",
                         "Packed GEMM timing excludes quantization; complete training timing includes conversion, quantization and backward.",
                         "Quantized gradients are stochastic; their difference from BF16 is not a kernel correctness failure."]}
    fatal = None
    with PowerMonitor() as monitor:
        try:
            torch.manual_seed(42)
            torch.set_num_threads(4)
            torch.backends.cuda.matmul.allow_tf32 = False
            report["gpu"] = torch.cuda.get_device_name()
            report["capability"] = torch.cuda.get_device_capability()
            {"kernel": kernel_probes, "training": training_probes, "grouped": grouped_probes,
             "causality": causal_scale_probe, "learning": learning_probe}[args.mode](report, args)
        except Exception as exc:
            fatal = repr(exc)
            report.update(fatal_error=fatal, traceback=traceback.format_exc())
    attach_power(report["cases"], monitor.samples)
    telemetry = args.out.with_suffix(".power.json")
    monitor.save(telemetry)
    report["power_samples"] = str(telemetry)
    report["passed"] = fatal is None and bool(report["cases"]) and all(c["status"] == "passed" for c in report["cases"])
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "passed": report["passed"], "fatal": fatal}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
