"""Experimental SM120 native W4A4 expert arithmetic.

Quartet-II supplies four-over-six forward quantization and randomized Hadamard
EDEN backward quantization. FlashInfer/CUTLASS supplies all three native FP4
GEMMs. This module owns no parameters, optimizer state or persistent caches.
Activation normalization is detached, per token and a power of two: no forward
quantization scale can depend on a later token or another resident expert.
"""

from collections import defaultdict
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@lru_cache(maxsize=8)
def _hadamard(device):
    from quartet2.linear import get_hadamard_matrix
    return get_hadamard_matrix(128, torch.bfloat16, device)


def row_scale(x):
    """Token-local detached power-of-two scale, including an exact zero case.

    Normalized finite BF16 rows have maximum magnitude < 2. Clamping the
    exponent keeps the scale representable even for BF16 subnormals and the
    largest finite BF16 value; it introduces no forward clipping bound.
    """
    maximum = x.detach().float().abs().amax(-1, keepdim=True)
    exponent = maximum.clamp_min(2.**-126).log2().floor().clamp(-126, 127)
    return torch.where(maximum == 0, torch.ones_like(maximum), torch.exp2(exponent))


def grouped_mm(left, right):
    """Multiply lists of quantized A[M_e,K] and B[N,K], with unequal alpha[e].

    Nonempty, 128-aligned operands are an internal contract. In particular,
    never launch the SM120 grouped kernel with an empty reduction dimension.
    CUTLASS's segmented SFA addressing includes gaps between expert segments;
    concatenating independently swizzled scale matrices alone is incorrect.
    """
    if not left or len(left) != len(right):
        raise ValueError("grouped FP4 GEMM needs matching nonempty operand lists")
    k = left[0].fp4.shape[1] * 2
    n = right[0].fp4.shape[0]
    counts = [a.fp4.shape[0] for a in left]
    if k == 0 or k % 128 or n == 0 or n % 128 or any(m == 0 or m % 128 for m in counts):
        raise ValueError("grouped FP4 GEMM dimensions must be positive multiples of 128")
    if any(a.fp4.shape[1] * 2 != k or b.fp4.shape != (n, k // 2)
           for a, b in zip(left, right)):
        raise ValueError("grouped FP4 GEMM requires common N and K")
    from flashinfer.gemm import group_gemm_nvfp4_nt_groupwise
    starts = [0]
    for m in counts:
        starts.append(starts[-1] + m)
    sf_starts = [(s + e * 127) // 128 * 128 for e, s in enumerate(starts[:-1])]
    sf = torch.zeros(sf_starts[-1] + counts[-1], k // 16,
                     device=left[0].fp4.device, dtype=torch.uint8)
    for start, a in zip(sf_starts, left):
        sf[start:start + a.fp4.shape[0]].copy_(a.micro_scales.view(torch.uint8))
    return group_gemm_nvfp4_nt_groupwise(
        torch.cat([a.fp4 for a in left]), torch.stack([b.fp4 for b in right]),
        sf, torch.stack([b.micro_scales.view(torch.uint8) for b in right]),
        torch.tensor(starts, device=sf.device, dtype=torch.int32),
        alpha=torch.stack([a.tensor_scale * b.tensor_scale for a, b in zip(left, right)]))


class _GroupedLinear(torch.autograd.Function):
    """Quartet's three-GEMM estimator, batched across accepted experts.

    Backward reductions may mix training examples; forward scales may not.
    Weight gradients are bucketed by padded token count because FlashInfer's
    grouped NT API has one K per call. Empty experts are removed by the caller.
    """

    @staticmethod
    def forward(ctx, x, weight, counts, had):
        from quartet2.quant import quant_fp4, NVFP4QuantMode
        bound = torch.full((), 2., device=x.device, dtype=torch.float32)
        qx = [quant_fp4(a.contiguous(), amax=bound, scale_override=1., mode=NVFP4QuantMode.FOUR_SIX)
              for a in x.split(counts)]
        # A separate maximum for each expert: changing the resident bank cannot
        # change the representation of an existing expert's weights.
        qw = [quant_fp4(w.to(torch.bfloat16).contiguous(), scale_override=1.,
                        mode=NVFP4QuantMode.FOUR_SIX) for w in weight.unbind()]
        ctx.counts, ctx.x_dtype, ctx.w_dtype = counts, x.dtype, weight.dtype
        ctx.save_for_backward(had, *(t for q in (*qx, *qw) for t in q))
        return grouped_mm(qx, qw)

    @staticmethod
    def backward(ctx, grad):
        from quartet2.linear import rerotate_hadamard
        from quartet2.quant import NVFP4Quant, rht128_quant_eden, rht128_requant
        saved = ctx.saved_tensors
        had = rerotate_hadamard(saved[0])[:16, :].contiguous()
        quantized = [NVFP4Quant(*saved[i:i + 3]) for i in range(1, len(saved), 3)]
        e = len(ctx.counts)
        qx, qw = quantized[:e], quantized[e:]
        scale = (17 / 16) * .93  # Qualified Quartet-II estimator, unchanged.
        eg, wt, et, xt = [], [], [], []
        for g, x, w in zip(grad.to(torch.bfloat16).split(ctx.counts), qx, qw):
            g = g.contiguous()
            eg.append(rht128_quant_eden(x=g, h=had, scale_override=scale))
            wt.append(rht128_requant(x=w.fp4, x_group_scales=w.micro_scales,
                                    x_tensor_scale=w.tensor_scale, h=had, scale_override=scale))
            et.append(rht128_quant_eden(x=g, h=had, scale_override=scale, transpose=True))
            xt.append(rht128_requant(x=x.fp4, x_group_scales=x.micro_scales,
                                    x_tensor_scale=x.tensor_scale, h=had, scale_override=scale))
        dx = grouped_mm(eg, wt)
        buckets = defaultdict(list)
        for i, count in enumerate(ctx.counts):
            buckets[count].append(i)
        dw = [None] * e
        for ids in buckets.values():
            result = grouped_mm([et[i] for i in ids], [xt[i] for i in ids])
            for i, chunk in zip(ids, result.split(et[ids[0]].fp4.shape[0])):
                dw[i] = chunk
        return dx.to(ctx.x_dtype), torch.stack(dw).to(ctx.w_dtype), None, None


def linear(x, weight, counts, had=None):
    """Padded expert-major linear with token-local forward normalization."""
    if x.device.type != "cuda" or x.dtype != torch.bfloat16:
        raise ValueError("native NVFP4 linear requires CUDA BF16 activations")
    if not counts or sum(counts) != x.shape[0] or weight.shape[0] != len(counts):
        raise ValueError("native NVFP4 linear counts must partition its input and weights")
    if any(c <= 0 or c % 128 for c in counts) or x.shape[1] % 128 or weight.shape[1] % 128:
        raise ValueError("native NVFP4 linear dimensions must be positive multiples of 128")
    if weight.shape[2] != x.shape[1] or x.shape[1] == 0 or weight.shape[1] == 0:
        raise ValueError("native NVFP4 linear weight shape must match positive input width")
    scale = row_scale(x)
    normalized = (x.float() / scale).to(torch.bfloat16)
    y = _GroupedLinear.apply(normalized, weight, tuple(counts),
                             _hadamard(x.device) if had is None else had)
    return (y.float() * scale).to(torch.bfloat16)


def experts(src, levels, expert_ids, tokens, scores, *, recompute):
    """Experimental accepted-route adapter; routing/paging remain caller owned."""
    from .precision import dispatch_dtype
    if (src.device.type != "cuda" or dispatch_dtype(src.device, src.dtype) != torch.bfloat16
            or len(levels) != 1 or torch.cuda.get_device_capability(src.device)[0] != 12):
        raise ValueError("nvfp4 expert backend requires SM120 CUDA BF16 compute and depth-1 experts")
    w1, w3, w2 = levels[0]
    if src.shape[1] % 128 or w1.shape[1] % 128:
        raise ValueError("nvfp4 expert backend requires widths divisible by 128")
    try:
        import quartet2.quant  # noqa: F401
        from flashinfer.gemm import group_gemm_nvfp4_nt_groupwise  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("nvfp4 backend needs optional Quartet-II and FlashInfer in the repository venv") from exc
    order = torch.argsort(expert_ids, stable=True)
    tokens, scores = tokens[order], scores[order]
    counts = torch.bincount(expert_ids, minlength=w1.shape[0]).tolist()
    active = [i for i, c in enumerate(counts) if c]
    logical = [counts[i] for i in active]
    padded = tuple((c + 127) // 128 * 128 for c in logical)
    ids = torch.tensor(active, device=src.device)

    def run(x, a, b, c):
        chunks = x[tokens].to(torch.bfloat16).split(logical)
        gathered = torch.cat([F.pad(chunk, (0, 0, 0, p - chunk.shape[0]))
                              for chunk, p in zip(chunks, padded)])
        a, b, c = (w.index_select(0, ids) for w in (a, b, c))
        gate = linear(gathered, a, padded)
        up = linear(gathered, b, padded)
        hidden = F.silu(gate) * up
        out = linear(hidden, c, padded)
        return torch.cat([chunk[:count] for chunk, count in zip(out.split(padded), logical)]).to(x.dtype)

    args = (src, w1, w3, w2)
    gathered = checkpoint(run, *args, use_reentrant=False) if recompute else run(*args)
    return torch.zeros_like(src).index_add(0, tokens, gathered * scores[:, None])
