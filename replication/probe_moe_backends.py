"""Compare complete BF16 expert forward/backward with external fixed routing.

Synthetic one-layer probe, not model throughput. Uses FP32 masters and gradients
and includes BF16 weight conversion/packing and each backend's dispatch work.
SonicMoE fusion may change rounding relative to mini-AGI's FP32 weighted sum.
"""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import traceback

import torch
import torch.nn.functional as F

from replication.gpu_monitor import PowerMonitor
from replication.probe_nvfp4 import attach_power, benchmark, error_metrics, print_case, provenance


def setup_case(tokens, experts=32, topk=8, active_experts=None):
    h, width = 512, 2048
    active = experts if active_experts is None else active_experts
    x = torch.randn(tokens, h, device="cuda", requires_grad=True)
    weights = [(torch.randn(experts, a, b, device="cuda") / b**.5).requires_grad_()
               for a, b in ((width, h), (width, h), (h, width))]
    # Generate per-token selections without replacement. Keep routing external
    # so this experiment does not substitute another model/router definition.
    probs = torch.linspace(1, .15, active, device="cuda").expand(tokens, -1)
    ids = torch.multinomial(probs, topk, replacement=False).int().flatten()
    token_ids = torch.arange(tokens, device="cuda", dtype=torch.int32).repeat_interleave(topk)
    scores = torch.softmax(torch.randn(tokens, topk, device="cuda"), -1).flatten().requires_grad_()
    return x, *weights, scores, token_ids, ids


def expert_forward(case, backend):
    x, w1, w3, w2, scores, token_ids, ids = case
    experts = w1.shape[0]
    xb = x.to(torch.bfloat16)
    if backend == "sonic":
        from sonicmoe import moe_general_routing_inputs
        from sonicmoe.enums import ActivationType
        packed = torch.stack((w1, w3), dim=2).flatten(1, 2).to(torch.bfloat16)
        y, _ = moe_general_routing_inputs(
            xb, scores, token_ids, ids, packed.permute(1, 2, 0), None,
            w2.to(torch.bfloat16).permute(1, 2, 0), None, experts,
            torch.cuda.current_stream().cuda_stream, ActivationType.SWIGLU)
        return y.float()

    order = torch.argsort(ids, stable=True)
    ei, ti = ids[order].long(), token_ids[order].long()
    src = xb[ti]
    a, b, c = (w.to(torch.bfloat16) for w in (w1, w3, w2))
    counts = torch.bincount(ei, minlength=experts)
    if backend == "native_grouped":
        # Compact empty experts rather than exposing zero-K backward groups.
        present = counts.nonzero().flatten()
        offs = counts[present].cumsum(0).int()
        gate = F.grouped_mm(src, a[present].transpose(-1, -2), offs=offs)
        up = F.grouped_mm(src, b[present].transpose(-1, -2), offs=offs)
        out = F.grouped_mm(F.silu(gate) * up, c[present].transpose(-1, -2), offs=offs)
    else:
        out, start = [], 0
        # Match the current exact dispatch's host counts and per-expert GEMMs.
        av, bv, cv = (w.unbind(0) for w in (a, b, c))
        for e, count in enumerate(counts.tolist()):
            if count:
                xe = src[start:start + count]
                out.append((F.silu(xe @ av[e].T) * (xe @ bv[e].T)) @ cv[e].T)
                start += count
        out = torch.cat(out)
    result = torch.zeros_like(x)
    result.index_add_(0, ti, out.float() * scores[order, None])
    return result


def run_case(report, tokens, active, args):
    torch.manual_seed(42)
    case = setup_case(tokens, active_experts=active)
    parameters = case[:5]  # x, three FP32 expert slabs, FP32 routing scores
    g = torch.randn_like(case[0])

    def step(backend):
        for p in parameters:
            p.grad = None
        y = expert_forward(case, backend)
        y.backward(g)
        return y

    expected = step("reference").detach().clone()
    gradients = [p.grad.clone() for p in parameters]
    for backend in args.backends:
        result = {"backend": backend, "tokens": tokens, "experts": 32,
                  "active_expert_pool": active, "topk": 8, "hidden": 512, "expert_width": 2048}
        report["cases"].append(result)
        try:
            y = step(backend)
            result["output_error"] = error_metrics(y, expected)
            result["gradient_errors"] = {
                name: error_metrics(p.grad, ref) for name, p, ref in
                zip(("input", "w1", "w3", "w2", "routing_scores"), parameters, gradients)}
            result["gradient_dtypes"] = [str(p.grad.dtype) for p in parameters]
            # Predeclared numerical acceptance for BF16 fused arithmetic;
            # bit-exact equivalence is not expected across different kernels.
            for metrics in [result["output_error"], *result["gradient_errors"].values()]:
                if not metrics["finite"] or metrics["relative_l2"] > .025:
                    raise AssertionError(f"BF16 candidate exceeds 2.5% relative L2 tolerance: {metrics}")
            if active < 32:
                result["empty_expert_gradients_zero"] = all(not bool(p.grad[active:].count_nonzero())
                                                            for p in parameters[1:4])
                if not result["empty_expert_gradients_zero"]:
                    raise AssertionError("unused expert has nonzero gradient")
            if args.seconds:
                result["complete_training_benchmark"] = benchmark(lambda: step(backend), tokens, args.seconds)
            result["status"] = "passed"
        except Exception as exc:
            result.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
        print_case(result)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tokens", type=int, nargs="+", default=[4096])
    ap.add_argument("--backends", nargs="+", default=["reference", "native_grouped", "sonic"])
    ap.add_argument("--seconds", type=float, default=3.)
    ap.add_argument("--empty-experts", action="store_true")
    args = ap.parse_args()
    if min(args.tokens) < 1 or args.seconds < 0 or any(b not in ("reference", "native_grouped", "sonic") for b in args.backends):
        ap.error("invalid tokens, seconds or backend")
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = {"configuration": vars(args) | {"out": str(args.out)}, "cases": [],
              **provenance(),
              "packages": {p: importlib.metadata.version(p) for p in ("torch", "triton", "quack-kernels", "sonic-moe")},
              "limits": ["Synthetic single expert layer with fixed external routing; no paging, attention, recurrence or optimizer timing.",
                         "Includes FP32-to-BF16 conversion and packing once per forward; a reusable packed-weight cache may change results.",
                         "SonicMoE returns BF16 output; reference weights expert outputs/sums in FP32, so rounding differs.",
                         "Predeclared acceptance: finite and relative L2 <= 2.5% for output and each gradient. Not a learning-quality guarantee."]}
    with PowerMonitor() as monitor:
        try:
            torch.set_num_threads(4)
            torch.backends.cuda.matmul.allow_tf32 = False
            report["gpu"] = torch.cuda.get_device_name()
            for tokens in args.tokens:
                run_case(report, tokens, 16 if args.empty_experts else 32, args)
        except Exception as exc:
            report.update(fatal_error=repr(exc), traceback=traceback.format_exc())
    attach_power(report["cases"], monitor.samples)
    telemetry = args.out.with_suffix(".power.json")
    monitor.save(telemetry)
    report["power_samples"] = str(telemetry)
    report["passed"] = "fatal_error" not in report and bool(report["cases"]) and all(c["status"] == "passed" for c in report["cases"])
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "passed": report["passed"]}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
