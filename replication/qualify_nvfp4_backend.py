"""Learning, estimator and complete-layer measurements for experimental NVFP4.

All GPU work, including startup and compilation, is power monitored. No model
files are loaded or written. Run as a module with the repository's Python.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import traceback
from unittest.mock import patch

import torch
import torch.nn.functional as F

from minagi import nvfp4
from minagi.expert_compute import compute_experts
from minagi.precision import amp, set_compute_dtype
from replication.gpu_monitor import PowerMonitor
from replication.probe_nvfp4 import (attach_power, benchmark, dequantize,
                                     error_metrics, provenance)


def gradients(args):
    x = torch.randn(384, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = (torch.randn(2, 2048, 512, device="cuda") / 512**.5).requires_grad_()
    g = torch.randn(384, 2048, device="cuda", dtype=torch.bfloat16)
    saved = []
    original = nvfp4.grouped_mm

    def record(a, b):
        saved.append((a, b))
        return original(a, b)

    with torch.no_grad(), patch.object(nvfp4, "grouped_mm", record):
        nvfp4.linear(x, w, (128, 256))
    scale = nvfp4.row_scale(x)
    refs = []
    dw_ref, dx_ref = [], []
    for (a, b), gs, ss in zip(zip(*saved[0]), g.split((128, 256)), scale.split((128, 256))):
        dx_ref.append(gs.float() @ dequantize(*b))
        dw_ref.append((gs.float() * ss).T @ dequantize(*a))
    dx_ref, dw_ref = torch.cat(dx_ref), torch.stack(dw_ref)
    dx, dw = torch.zeros_like(x, dtype=torch.float32), torch.zeros_like(w)
    for i in range(args.samples):
        y = nvfp4.linear(x, w, (128, 256))
        a, b = torch.autograd.grad(y, (x, w), g)
        dx += a.float()
        dw += b
        if i in (0, 7, 31, args.samples - 1):
            refs.append({"samples": i + 1,
                         "input": error_metrics(dx / (i + 1), dx_ref),
                         "weight": error_metrics(dw / (i + 1), dw_ref)})
    if not all(m["finite"] and m["relative_l2"] < .06 for m in refs[-1].values() if isinstance(m, dict)):
        raise AssertionError(f"Mean gradient differs from decoded-forward STE reference: {refs[-1]}")
    return {"shape_emkn": [2, [128, 256], 512, 2048], "mean_gradient_errors": refs,
            "limits": "Finite Monte Carlo check, not proof of unbiasedness; compares to decoded forward operands."}


def learning(args):
    # Equal seeds, batches, initialization, optimizer and heldout deployment.
    e, h, n, total, train, batch = 2, 128, 256, 2048, 1536, 128
    inputs = torch.randn(e, total, h, device="cuda")
    inputs *= torch.tensor([.25, 4.], device="cuda")[:, None, None]
    teacher = torch.randn(e, n, h, device="cuda") / h**.5
    targets = inputs @ teacher.transpose(1, 2)
    initial = torch.randn_like(teacher) / h**.5
    generator = torch.Generator().manual_seed(314)
    batches = [torch.randint(train, (batch,), generator=generator).to("cuda") for _ in range(args.steps)]
    result = []
    for backend in ("bf16", "nvfp4"):
        torch.manual_seed(2718)
        w = torch.nn.Parameter(initial.clone())
        opt = torch.optim.AdamW([w], lr=.003, weight_decay=0., fused=True)

        def predict(x, deployed=False):
            xb = x.to(torch.bfloat16)
            if backend == "nvfp4" or deployed:
                return nvfp4.linear(xb.flatten(0, 1), w, (x.shape[1],) * e).view(e, x.shape[1], n)
            return torch.bmm(xb, w.to(torch.bfloat16).transpose(1, 2))

        curve = []
        for step in range(args.steps + 1):
            if step % 50 == 0 or step == args.steps:
                with torch.no_grad():
                    actual = predict(inputs[:, train:]).float()
                    curve.append({"step": step, "heldout_mse": float(F.mse_loss(actual, targets[:, train:])),
                                  "per_expert_mse": [float(F.mse_loss(a, b)) for a, b in zip(actual, targets[:, train:])]})
                print(json.dumps({"backend": backend, **curve[-1]}), flush=True)
            if step == args.steps:
                break
            opt.zero_grad(set_to_none=True)
            ids = batches[step]
            loss = F.mse_loss(predict(inputs[:, ids]).float(), targets[:, ids])
            loss.backward()
            if not bool(torch.isfinite(loss) & torch.isfinite(w.grad).all()):
                raise AssertionError("non-finite learning loss or master gradient")
            opt.step()
        with torch.no_grad():
            deployed = float(F.mse_loss(predict(inputs[:, train:], deployed=True).float(), targets[:, train:]))
        if curve[-1]["heldout_mse"] >= .25 * curve[0]["heldout_mse"]:
            raise AssertionError("Heldout regression error did not fall by 75%")
        result.append({"backend": backend, "curve": curve, "heldout_nvfp4_mse": deployed,
                       "master_dtype": str(w.dtype)})
    return {"cases": result, "steps": args.steps, "shape": [e, batch, h, n],
            "limits": "Untuned two-expert linear regression; no language-model quality or QAT advantage claim."}


def layer_benchmark(args):
    t, e, d, f, k = 4096, 32, 512, 2048, 8
    x = torch.randn(t, d, device="cuda", requires_grad=True)
    weights = tuple((torch.randn(*s, device="cuda") / s[-1]**.5).requires_grad_()
                    for s in ((e, f, d), (e, f, d), (e, d, f)))
    # Unique experts per token, with a reproducible skew (one unused expert).
    expert_ids = torch.rand(t, e - 1, device="cuda").pow(
        torch.linspace(.5, 2., e - 1, device="cuda")).topk(k, dim=1).indices.flatten()
    tokens = torch.arange(t, device="cuda").repeat_interleave(k)
    scores = torch.full((t * k,), 1 / k, device="cuda", requires_grad=True)
    g = torch.randn_like(x) / (t * d)
    results = []
    # ABBA comparisons for each control, all operands and gradients included.
    for backend in ("reference", "nvfp4", "nvfp4", "reference", "sonic", "nvfp4", "nvfp4", "sonic"):
        def step():
            for p in (x, *weights, scores):
                p.grad = None
            with amp("cuda"):
                y = compute_experts(x, [weights], expert_ids, tokens, scores, backend=backend)
            y.backward(g)
        measured = benchmark(step, t, args.seconds)
        for p in (x, *weights, scores):
            if not bool(torch.isfinite(p.grad).all()):
                raise AssertionError(f"non-finite {backend} gradient")
        measured["backend"] = backend
        results.append(measured)
        print(json.dumps({"backend": backend, "median_seconds": measured["median_seconds"]}), flush=True)
    return {"cases": results, "shape_tokens_experts_hidden_width_topk": [t, e, d, f, k],
            "accepted_counts": torch.bincount(expert_ids, minlength=e).tolist(),
            "limits": "Synthetic fixed routes, full expert layer forward/backward; excludes recurrent model and optimizer."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("gradients", "learning", "benchmark"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seconds", type=float, default=3.)
    args = parser.parse_args()
    if args.samples < 1 or args.steps < 1 or args.seconds <= 0:
        parser.error("samples, steps and seconds must be positive")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = {"configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "provenance": provenance(),
              "sources_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                 (Path(__file__), Path(nvfp4.__file__), Path("minagi/expert_compute.py"))}}
    report["dependencies"] = {name: importlib.metadata.version(name) for name in
                              ("torch", "quartet2", "flashinfer-python", "sonic-moe", "quack-kernels")}
    with PowerMonitor() as monitor:
        try:
            torch.manual_seed(42)
            torch.set_num_threads(1)
            torch.backends.cuda.matmul.allow_tf32 = False
            set_compute_dtype("bf16")
            report["result"] = {"gradients": gradients, "learning": learning,
                                "benchmark": layer_benchmark}[args.mode](args)
            torch.cuda.synchronize()
            report["status"] = "passed"
        except BaseException as exc:
            report.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
            raise
        finally:
            # The last sample brackets every measured interval.
            monitor.sample()
            attach_power(report, monitor.samples)
            monitor.save(args.out.with_suffix(".power.json"))
            args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
