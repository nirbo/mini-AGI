"""Short, matched BF16 checkpoint adaptation screen; all updates stay in RAM.

Uses disjoint windows from the locally available TinyStories validation text
as adaptation/tuning/test partitions. This is not a canonical benchmark split
or a pretraining convergence result. Both optimizers start with fresh history.
"""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import types

os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'

import numpy as np
import torch
import torch.nn.functional as F

from minagi.ingest import as_stream
from minagi.optimizers import make_optimizer
from minagi.paged import read_npz
from minagi.pool import PooledMLP
from minagi.precision import amp, set_compute_dtype
from minagi.recur import load_any
from replication.gpu_monitor import PowerMonitor, summarize_samples
from train import _split_trunk_pool
from replication.muon_candidates import ResearchPagedMatrix, VARIANTS, SOURCES, matrix_optimizer


def fingerprint(tree):
    digest=hashlib.sha256()
    def visit(value):
        if torch.is_tensor(value):
            digest.update(str((str(value.dtype),tuple(value.shape))).encode())
            digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value,np.ndarray):
            digest.update(str((str(value.dtype),value.shape)).encode())
            digest.update(value.tobytes())
        elif isinstance(value,dict):
            for key in sorted(value,key=repr):
                digest.update(repr(key).encode())
                visit(value[key])
        elif isinstance(value,(tuple,list)):
            digest.update(type(value).__name__.encode())
            for item in value: visit(item)
        else:
            digest.update(repr(value).encode())
    visit(tree)
    return digest.hexdigest()


def sonic_autotuning_configs():
    from triton.runtime.autotuner import Autotuner
    result={}
    for name,module in list(sys.modules.items()):
        if not name.startswith('sonicmoe.') or module is None:
            continue
        for attribute,value in vars(module).items():
            if isinstance(value,Autotuner):
                result[name+'.'+attribute]={repr(k):dict(kwargs=c.kwargs,num_warps=c.num_warps,num_stages=c.num_stages)
                                               for k,c in value.cache.items()}
    return result


def pin_sonic_reduction():
    """Freeze the existing, measured Triton reduction config for this screen.

    Torch's deterministic flag does not govern third-party autotuner choices.
    This changes process-local dispatch only, not the installed package.
    """
    from sonicmoe.functional.reduction_over_k_gather import token_gather_sum_kernel
    from triton import Config
    config=Config({'BLOCK_H':256,'BLOCK_K':32},num_warps=4,num_stages=4)
    token_gather_sum_kernel.configs=[config]
    token_gather_sum_kernel.cache.clear()
    return dict(kwargs=config.kwargs,num_warps=config.num_warps,num_stages=config.num_stages)


def weights_only(tiers, uid):
    # Fresh history for BOTH candidates, including experts paged in later.
    return {k: torch.from_numpy(v) for k, v in read_npz(tiers._file(uid), ("w1", "w3", "w2")).items()}


def tensor_bytes(tree):
    seen = set()
    def visit(value):
        if torch.is_tensor(value):
            key = (value.device, value.untyped_storage().data_ptr())
            if key not in seen:
                seen.add(key)
                return value.untyped_storage().nbytes()
        elif isinstance(value, dict):
            return sum(visit(v) for v in value.values())
        elif isinstance(value, (list, tuple)):
            return sum(visit(v) for v in value)
        return 0
    return visit(tree)


@torch.no_grad()
def evaluate(model, data, offset, context, windows):
    model.eval()
    losses = []
    for i in range(windows):
        start = offset + i * context
        ids = torch.tensor(data[start:start + context + 1].astype(np.int64), device="cuda")[None]
        with amp("cuda"):
            logits, _ = model(ids[:, :-1])
        losses.append(float(F.cross_entropy(logits.float().flatten(0, 1), ids[:, 1:].flatten())))
    return {"mean_ce": float(np.mean(losses)), "per_window_ce": losses}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--optimizer", choices=("adamw", "gefen", *VARIANTS), required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--aux-lr", type=float, default=1e-4)
    parser.add_argument('--fingerprint',action='store_true',help='Hash gradients/weights/history for reproducibility probes; adds overhead.')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--eval-windows", type=int, default=16)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not (args.context > 0 and 0 < args.steps <= (262144 - 1) // args.context and args.depth > 0
            and args.eval_windows > 0 and args.eval_windows * args.context < 262144):
        parser.error("invalid dimensions or overlapping data partitions")
    report = {"configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "sources": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                          for name in ("minagi/optimizers.py", "minagi/optim_codec.py", "replication/muon_candidates.py", __file__)},
              "upstream_sources": SOURCES,
              "limits": ["Short adaptation on disjoint portions of one validation text, not a canonical train/test split or convergence study.",
                         "Fixed depth 6 by default, versus the checkpoint's depth 24; only Sonic BF16 compute is tested.",
                         "Fresh optimizer history for both candidates; checkpoint Adam moments are deliberately ignored.",
                         "Every expert touched remains cached in host RAM; disk writes are disabled. Realistic bounded host-cache performance needs separate measurement.",
                         "LR screening/confirmation is limited to the requested grid; power includes desktop activity and first-step calibration."]}
    with PowerMonitor() as monitor:
        try:
            torch.use_deterministic_algorithms(True)
            torch.set_num_threads(4)
            torch.manual_seed(args.seed)
            torch.cuda.set_per_process_memory_fraction(.75)
            torch.backends.cuda.matmul.allow_tf32 = False
            set_compute_dtype("bf16")
            model, _ = load_any(str(args.weights), torch.device("cuda"), read_only=True)
            model.cfg.max_steps = model.cfg.min_steps = model.cfg.bptt_window = args.depth
            model.cfg.train_steps_mean = 0
            for site in model.modules():
                if isinstance(site, PooledMLP):
                    site.expert_backend = "sonic"
                    site.grad_checkpoint = True
            pool = model.pool
            if model.cfg.d_model!=512 or pool.resident!=32:
                raise ValueError('fixed reduction is qualified only for hidden 512 and 32 resident experts')
            report['fixed_sonic_reduction']=pin_sonic_reduction()
            # Compile the optimizer on synthetic weights, with no checkpoint
            # updates. Record this separately from learning time and preserve
            # the model/data RNG. All warmup remains externally power monitored.
            if args.optimizer in VARIANTS:
                warm_start = time.perf_counter()
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    ps = [torch.nn.Parameter(torch.randn_like(getattr(pool, n)[s]))
                          for s in range(pool.resident) for n in ("w1", "w3", "w2")]
                    warm_opt = matrix_optimizer(args.optimizer, ps, args.lr)
                    for p in ps:
                        p.grad = torch.randn_like(p) * .001
                    for _ in range(2):
                        warm_opt.step()
                    torch.cuda.synchronize()
                    del warm_opt, ps, p
                gc.collect()
                torch.cuda.empty_cache()
                report['synthetic_optimizer_warmup_seconds'] = time.perf_counter() - warm_start
            if pool.tiers.ram or any(e >= 0 for e in pool.slots):
                raise AssertionError("fresh-history screen requires an initially empty pool")
            pool.tiers.ram_capacity = pool.n_experts()
            pool.tiers._from_disk = types.MethodType(weights_only, pool.tiers)
            data = as_stream(str(args.text))
            if len(data) < 524288 + args.eval_windows * args.context + 1:
                raise ValueError("text does not cover the reserved evaluation partitions")
            report["manifest_sha256"] = hashlib.sha256((args.weights / "manifest.json").read_bytes()).hexdigest()
            report["text_sha256"] = hashlib.sha256(args.text.read_bytes()).hexdigest()
            report["initial_tune"] = evaluate(model, data, 262144, args.context, args.eval_windows)
            report["initial_test"] = evaluate(model, data, 524288, args.context, args.eval_windows)
            trunk, experts = _split_trunk_pool(model)
            slabs = [getattr(pool, n) for n in ("w1", "w3", "w2")]
            auxiliary = [p for p in experts if all(p is not slab for slab in slabs)]
            groups = [{"params": trunk, "name": "trunk", "lr": args.aux_lr * .2},
                      {"params": auxiliary, "name": "pool_aux", "lr": args.aux_lr},
                      {"params": slabs, "name": "pool", "lr": args.lr}]
            if args.optimizer in VARIANTS:
                opt = ResearchPagedMatrix(groups, pool, args.optimizer, args.lr,
                                          betas=(.9, .95), weight_decay=.1, fused=True)
            else:
                opt = make_optimizer(groups, pool=pool, optimizer=args.optimizer, lr=args.lr,
                                     betas=(.9, .95), weight_decay=.1, fused=True)
            pool.attach_optimiser(opt)
            if hasattr(opt, "_gefen"):
                # A GF-trained source may have an immutable durable codebook.
                # This weights-only experiment intentionally resets it too.
                assert pool.tiers.read_only
                opt._gefen._gefen_codebook = None
                opt._gefen._gefen_global_step = 0
                opt._durable_codebook_id = None
            indices = np.random.default_rng(args.seed).permutation((262144 - 1) // args.context)[:args.steps]
            report["training_window_indices"] = indices.tolist()
            trace = []
            model.train()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            for tick, index in enumerate(indices):
                t0 = time.perf_counter()
                offset = int(index) * args.context
                ids = torch.tensor(data[offset:offset + args.context + 1].astype(np.int64), device="cuda")[None]
                opt.zero_grad(set_to_none=True)
                with amp("cuda"):
                    _, loss = model(ids[:, :-1], ids[:, 1:])
                    loss = loss + model.cfg.pool_aux * model.pool_aux() + model.pool_balance()
                loss.backward()
                if args.fingerprint and tick==0:
                    report['first_gradient_sha256']=fingerprint({n:p.grad for n,p in model.named_parameters()})
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(grad_norm)):
                    raise AssertionError("non-finite training objective/gradient norm")
                torch.cuda.synchronize()
                opt_start = time.perf_counter()
                opt.step()
                torch.cuda.synchronize()
                trace.append({"step": tick + 1, "objective": float(loss.detach()),
                              "gradient_norm_before_clip": float(grad_norm),
                              'routing_seed':pool._rng_seed,
                              "seconds": time.perf_counter() - t0,
                              "optimizer_seconds": time.perf_counter() - opt_start})
                if (tick + 1) % 16 == 0:
                    print(json.dumps(trace[-1]), flush=True)
            end = time.perf_counter()
            report.update(training_start=start, training_end=end, trace=trace,
                          peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                          adam_state_bytes=tensor_bytes(dict(opt.state)))
            if hasattr(opt, "_gefen"):
                report["resident_gefen_state_bytes"] = tensor_bytes([dict(opt._gefen.state), opt._gefen._gefen_codebook])
            if hasattr(opt, "_matrix"):
                report["resident_matrix_state_bytes"] = tensor_bytes(dict(opt._matrix.state))
                report["host_matrix_state_bytes"] = tensor_bytes(opt._history)
                report["expert_projection_update_counts"] = {str(k): v for k, v in opt._counts.items()}
            opt.zero_grad(set_to_none=True)
            report["final_tune"] = evaluate(model, data, 262144, args.context, args.eval_windows)
            report["final_test"] = evaluate(model, data, 524288, args.context, args.eval_windows)
            if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
                raise AssertionError("non-finite trained parameter")
            if pool.tiers.writebacks or pool.tiers.evictions or not pool.tiers.read_only:
                raise AssertionError("screen must retain every touched expert without disk writes")
            if args.fingerprint:
                report['final_model_sha256']=fingerprint(dict(model.state_dict()))
                report['final_optimizer_sha256']=fingerprint(opt.state_dict())
                report['final_host_experts_sha256']=fingerprint(dict(pool.tiers.ram))
                report['final_slots']=pool.slots
            report['deterministic_algorithms_enabled']=torch.are_deterministic_algorithms_enabled()
            report['sonic_autotuning_configs']=sonic_autotuning_configs()
            report["status"] = "passed"
        except BaseException as exc:
            report.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
            raise
        finally:
            monitor.sample()
            if "training_end" in report:
                report["training_power"] = summarize_samples(monitor.samples, report["training_start"],
                                                               report["training_end"], args.steps * args.context)
            monitor.save(args.out.with_suffix(".power.json"))
            args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "initial_tune", "final_tune", "final_test", "peak_allocated_gib", "training_power")}), flush=True)


if __name__ == "__main__":
    main()
