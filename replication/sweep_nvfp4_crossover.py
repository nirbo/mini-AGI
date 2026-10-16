"""Bounded complete expert-layer training sweep against Sonic, with full state.

Run one case/backend per process. All CUDA work is power monitored. The process
allocator budget leaves desktop headroom; no hardware configuration is changed.
Parameters and AdamW state are synthetic and never saved as model artifacts.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import traceback

import torch

from minagi.expert_compute import compute_experts
from minagi.paged import _CachedCast
from minagi.precision import amp, set_compute_dtype
from replication.gpu_monitor import PowerMonitor, summarize_samples
from replication.probe_nvfp4 import benchmark


def routes(experts, rows_per_expert, topk, device):
    if min(experts, rows_per_expert, topk) <= 0 or topk > experts:
        raise ValueError("positive expert/row counts and topk <= experts required")
    if experts * rows_per_expert % topk:
        raise ValueError("assignments must divide evenly into input tokens")
    tokens = experts * rows_per_expert // topk
    t = torch.arange(tokens, device=device)
    ids = ((t[:, None] * topk + torch.arange(topk, device=device)) % experts).flatten()
    return ids, t.repeat_interleave(topk)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--experts", type=int, required=True)
    parser.add_argument("--rows-per-expert", type=int, required=True)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--backend", choices=("sonic", "nvfp4"), required=True)
    parser.add_argument("--budget-gib", type=float, default=24.)
    parser.add_argument("--seconds", type=float, default=3.)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if min(args.hidden, args.width, args.experts, args.rows_per_expert, args.topk) <= 0:
        parser.error("dimensions must be positive")
    if args.hidden % 128 or args.width % 128 or args.rows_per_expert % 128:
        parser.error("matrix dimensions must be divisible by 128")
    if args.topk > args.experts or args.budget_gib <= 0 or args.seconds <= 0:
        parser.error("invalid topk, budget or seconds")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    paths = (Path(__file__), Path("minagi/nvfp4.py"), Path("minagi/expert_compute.py"))
    report = {"configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              "limits": ["Balanced fixed accepted routes, complete expert-layer forward/backward plus fused AdamW update.",
                         "Includes persistent FP32 weights/gradients/moments and BF16 compute cache, refreshed after every update.",
                         "Excludes attention, recurrent unrolling, embedding/head, paging and model checkpointing. Not a full-model fit claim.",
                         "Synthetic linear output objective is a performance workload, not a learning-quality experiment.",
                         "Per-process PyTorch allocator cap does not cover every third-party CUDA allocation; additional free-memory margin is retained."]}
    with PowerMonitor() as monitor:
        try:
            torch.manual_seed(42)
            torch.set_num_threads(4)
            torch.backends.cuda.matmul.allow_tf32 = False
            set_compute_dtype("bf16")
            free, total = torch.cuda.mem_get_info()
            budget = min(args.budget_gib * 2**30, free - 4 * 2**30, total * .85)
            if budget <= 0:
                raise RuntimeError("insufficient free memory to preserve desktop headroom")
            torch.cuda.set_per_process_memory_fraction(budget / total)
            report.update(allocator_budget_gib=budget / 2**30, initially_free_gib=free / 2**30)
            e, d, f = args.experts, args.hidden, args.width
            p = 3 * e * d * f
            report.update(resident_parameters=p, persistent_state_and_cache_gib=18 * p / 2**30)
            if 18 * p + 2 * 2**30 > budget:
                report.update(status="skipped_budget", reason="Persistent state/cache alone leaves less than 2 GiB for layer temporaries")
            else:
                ids, tokens = routes(e, args.rows_per_expert, args.topk, "cuda")
                t = e * args.rows_per_expert // args.topk
                x = torch.randn(t, d, device="cuda", requires_grad=True)
                weights = tuple(torch.nn.Parameter(torch.empty(s, device="cuda").normal_(std=s[-1]**-.5))
                                for s in ((e, f, d), (e, f, d), (e, d, f)))
                scores = torch.full((tokens.numel(),), 1 / args.topk, device="cuda", requires_grad=True)
                upstream = torch.randn_like(x) / (t * d)
                optimizer = torch.optim.AdamW(weights, lr=1e-5, betas=(.9, .95), weight_decay=.1, fused=True)
                cache = tuple(w.detach().to(torch.bfloat16) for w in weights)

                def step():
                    optimizer.zero_grad(set_to_none=True)
                    x.grad = scores.grad = None
                    compute = tuple(_CachedCast.apply(w, c) for w, c in zip(weights, cache))
                    with amp("cuda"):
                        y = compute_experts(x, [compute], ids, tokens, scores, backend=args.backend)
                    y.backward(upstream)
                    optimizer.step()
                    # Preserve allocated immutable cache data per generation;
                    # after backward no previous graph refers to these buffers.
                    with torch.no_grad():
                        for c, w in zip(cache, weights):
                            c.copy_(w)

                measured = benchmark(step, t, args.seconds)
                for value in (x.grad, scores.grad, *(w.grad for w in weights), *weights):
                    if not bool(torch.isfinite(value).all()):
                        raise AssertionError("non-finite synthetic weight/gradient")
                allocated_states = sum(v.numel() * v.element_size() for state in optimizer.state.values()
                                       for v in state.values() if torch.is_tensor(v))
                report.update(status="passed", input_token_rows=t, accepted_assignments=t * args.topk,
                              optimizer_state_gib=allocated_states / 2**30, measurement=measured,
                              peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
        except torch.OutOfMemoryError as exc:
            report.update(status="out_of_memory", error=str(exc))
            gc.collect()
        except BaseException as exc:
            report.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
            raise
        finally:
            monitor.sample()
            if "measurement" in report:
                m = report["measurement"]
                m["power"] = summarize_samples(monitor.samples, m["start_monotonic"], m["end_monotonic"],
                                                new_tokens=m["token_rows"])
            monitor.save(args.out.with_suffix(".power.json"))
            args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "resident_parameters", "persistent_state_and_cache_gib")
                      if k in report}), flush=True)


if __name__ == "__main__":
    main()
