"""Expert arithmetic over already accepted routes.

Routing, admission, capacity, statistics and weight ownership belong to the
caller. Backends receive slot-indexed weight levels and token-major accepted
assignments, and return the weighted mixture in the residual stream's dtype.
There is no backend-specific state in checkpoints. The reference supports all
existing precisions/depths; SonicMoE is an optional BF16, single-level backend.
"""

import itertools
import os

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .precision import dispatch_dtype


def backend_name(override=None):
    name = (override if override is not None else
            os.environ.get("MINAGI_EXPERT_BACKEND", "reference")).strip().lower()
    if name not in ("reference", "sonic"):
        raise ValueError(f"unknown expert backend {name!r}; expected reference or sonic")
    return name


def _few(src, dt, weights, runs, tokens):
    xs = src[tokens].to(dt)
    w1, w3, w2 = weights
    outputs, start = [], 0
    for expert, count in runs:
        xe = xs[start:start + count]
        outputs.append((F.silu(xe @ w1[expert].T) * (xe @ w3[expert].T)) @ w2[expert].T)
        start += count
    return torch.cat(outputs).to(src.dtype)


def _reference(src, levels, experts, tokens, scores, *, recompute, padded, few):
    order = torch.argsort(experts, stable=True)
    experts, tokens, scores = experts[order], tokens[order], scores[order]
    n, d = levels[0][0].shape[0], src.shape[1]
    dt = dispatch_dtype(src.device, src.dtype)
    if few and experts.numel() <= 64 and len(levels) == 1:
        runs = [(e, sum(1 for _ in group)) for e, group in itertools.groupby(experts.tolist())]
        gathered = _few(src, dt, levels[0], runs, tokens)
    else:
        counts = torch.bincount(experts, minlength=n)
        counts_list = counts.tolist()
        cap = max(counts_list)
        starts = counts.cumsum(0) - counts
        slots = torch.arange(experts.numel(), device=src.device) - starts[experts]
        runs = [(e, c) for e, c in enumerate(counts_list) if c]

        def exact(x, w1, w3, w2):
            xs = x[tokens].to(dt)
            a, b, c = (w.to(dt).unbind(0) for w in (w1, w3, w2))
            outputs, start = [], 0
            for expert, count in runs:
                xe = xs[start:start + count]
                outputs.append((F.silu(xe @ a[expert].T) * (xe @ b[expert].T)) @ c[expert].T)
                start += count
            return torch.cat(outputs).to(x.dtype)

        def rectangle(x, *weights):
            lv = [weights[i:i + 3] for i in range(0, len(weights), 3)]
            buf = torch.zeros(n, cap, d, device=x.device, dtype=dt)
            buf[experts, slots] = x[tokens].to(dt)
            if len(lv) == 1:
                w1, w3, w2 = lv[0]
                h = F.silu(torch.bmm(buf, w1.transpose(1, 2).to(dt))) * \
                    torch.bmm(buf, w3.transpose(1, 2).to(dt))
                y = torch.bmm(h, w2.transpose(1, 2).to(dt))
            else:
                y = buf
                for w1, w3, w2 in lv:
                    h = F.silu(torch.bmm(y, w1.transpose(1, 2).to(dt))) * \
                        torch.bmm(y, w3.transpose(1, 2).to(dt))
                    y = y + torch.bmm(h, w2.transpose(1, 2).to(dt))
            return y[experts, slots].to(x.dtype)

        fn = rectangle if padded or len(levels) != 1 else exact
        weights = tuple(w for level in levels for w in level)
        gathered = (checkpoint(fn, src, *weights, use_reentrant=False) if recompute
                    else fn(src, *weights))
    out = torch.zeros_like(src)
    out.index_add_(0, tokens, gathered * scores.unsqueeze(-1))
    return out


def _sonic(src, levels, experts, tokens, scores, *, recompute):
    dt = dispatch_dtype(src.device, src.dtype)
    if src.device.type != "cuda" or dt != torch.bfloat16 or len(levels) != 1:
        raise ValueError("sonic expert backend requires CUDA BF16 compute and depth-1 experts; "
                         "use MINAGI_EXPERT_BACKEND=reference for other configurations")
    w1, w3, w2 = levels[0]
    if src.shape[1] % 16 or w1.shape[1] % 16:
        raise ValueError("sonic expert backend requires hidden and expert widths divisible by 16")
    if torch.cuda.get_device_capability(src.device)[0] not in (9, 10, 12):
        raise ValueError("sonic expert backend requires a QuACK-supported SM90/SM100/SM120 GPU family")
    try:
        from sonicmoe import moe_general_routing_inputs
        from sonicmoe import functional as sonic_functional
        from sonicmoe.enums import ActivationType
    except ImportError as exc:
        raise RuntimeError("sonic expert backend needs the optional SonicMoE/QuACK packages "
                           "in the repository venv; see replication/NVFP4_FEASIBILITY.md") from exc
    if not getattr(sonic_functional, "_MINAGI_FIXED_GEMM_CONFIG", False):
        raise RuntimeError("sonic expert backend requires the qualified source patch "
                           "replication/patches/sonicmoe-fixed-gemm-config.patch; "
                           "see replication/EXPERT_BACKENDS.md")
    # A backward grouped GEMM reduces over each expert's accepted rows. Never
    # pass an empty reduction group to the native kernels: remove unused
    # slots, retaining an autograd scatter back to the pool-owned slabs.
    present, expert_ids = torch.unique(experts, sorted=True, return_inverse=True)
    if present.numel() != w1.shape[0]:
        w1, w3, w2 = (w.index_select(0, present) for w in (w1, w3, w2))
    # Accepted routes arrive in token order, even with variable capacity drops.
    # Never substitute the library's router: admission and expert gates have
    # already been applied. Bind metadata and casts before checkpoint replay.
    token_ids, expert_ids = tokens.to(torch.int32).contiguous(), expert_ids.to(torch.int32)

    def run(x, a, b, c, route_scores):
        # Packing remains per use, inside the recomputed region. No shared
        # differentiable cache may accumulate gradients in BF16 across uses.
        packed = torch.stack((a.to(dt), b.to(dt)), dim=2).flatten(1, 2)
        out, _ = moe_general_routing_inputs(
            x.to(dt), route_scores.float(), token_ids, expert_ids,
            packed.permute(1, 2, 0), None, c.to(dt).permute(1, 2, 0), None,
            a.shape[0], torch.cuda.current_stream(x.device).cuda_stream,
            ActivationType.SWIGLU, is_inference_mode_enabled=not torch.is_grad_enabled())
        return out.to(x.dtype)

    args = (src, w1, w3, w2, scores)
    return checkpoint(run, *args, use_reentrant=False) if recompute else run(*args)


def compute_experts(src, levels, experts, tokens, scores, *, backend="reference",
                    recompute=False, padded=False, few=False):
    """Compute an accepted sparse mixture; token indices must be ascending.

    ``levels`` contains (gate/up/down) slabs [E,F,D], [E,F,D], [E,D,F].
    Each assignment supplies one slot ID, input token ID and differentiable
    score (including the expert gate). Zero-score assignments remain present:
    their score/gate derivatives can be nonzero. Empty assignment lists return
    zeros, matching the historical dispatch. Callers bind immutable compute
    generations before entry, so paging cannot replace replay's weight data.
    """
    backend = backend_name(backend)
    if not experts.numel():
        return torch.zeros_like(src)
    if backend == "sonic":
        return _sonic(src, levels, experts, tokens, scores, recompute=recompute)
    return _reference(src, levels, experts, tokens, scores,
                      recompute=recompute, padded=padded, few=few)
