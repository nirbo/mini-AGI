"""Research-only ordinary FP8 W8A8 expert arithmetic with FP32 masters.

E4M3 rowwise weights/activations; E5M2 backward signals; FP32 GEMM output
and gradient accumulation. Quantization is an approximate training estimator,
not the derivative of rounding. QuACK performs native FP8 batched GEMMs.
Routing, paging, checkpoints on disk and production backend choices are untouched.
"""

from contextlib import contextmanager
import math
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@torch.compile(fullgraph=True, dynamic=True, recompile_limit=64)
def quantize_rows(x, dtype=torch.float8_e4m3fn):
    maximum = torch.finfo(dtype).max
    values = x.float()
    scale = (values.abs().amax(-1, keepdim=True) / maximum).clamp_min(torch.finfo(torch.float32).tiny)
    quantized = (values / scale).clamp(-maximum, maximum).to(dtype)
    return quantized.contiguous(), scale.contiguous()


@torch.compile(fullgraph=True, dynamic=True, recompile_limit=64)
def rescale(out, sa, sb, dtype):
    return (out * sa * sb.mT).to(dtype)


def native_mm(a, sa, b, sb, *, out_dtype=torch.bfloat16, engine='quack'):
    """(row-quantized A) @ (row-quantized B).T; scales are ordinary FP32."""
    if engine == 'torch':
        if a.ndim != 2:
            raise ValueError('Torch comparison engine accepts a single 2D GEMM')
        return torch._scaled_mm(a, b.T, scale_a=sa, scale_b=sb.T.contiguous(),
                                out_dtype=out_dtype, use_fast_accum=False)
    if engine != 'quack':
        raise ValueError(f'unknown FP8 GEMM engine: {engine}')
    from quack.gemm_interface import gemm
    # Unit hardware block scales inside QuACK are an instruction selection
    # detail. Stored data are plain FP8 with per-row FP32 scales, not MXFP8.
    out = gemm(a, b.mT, out_dtype=torch.float32, tuned=False)
    return rescale(out, sa, sb, out_dtype)


class WeightCache:
    """One immutable quantized generation; replay closures retain older data."""

    def __init__(self):
        self.key = None
        self.weights = None
        self.conversions = 0
        self.hits = 0

    def clear(self):
        self.key = self.weights = None

    @torch.no_grad()
    def bind(self, weights, generation=None):
        key = (generation, tuple((id(w), w.data_ptr(), w._version, w.device, w.dtype,
                                  tuple(w.shape), w.stride()) for w in weights))
        if self.key != key:
            self.weights = tuple((*quantize_rows(w.detach()),
                                  *quantize_rows(w.detach().mT.contiguous())) for w in weights)
            self.key = key
            self.conversions += 1
        else:
            self.hits += 1
        return self.weights


class _Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, master, qw, sw, qwt, swt, engine):
        ctx.engine, ctx.master_dtype, ctx.input_dtype = engine, master.dtype, x.dtype
        ctx.save_for_backward(x, qwt, swt)
        qx, sx = quantize_rows(x)
        return native_mm(qx, sx, qw, sw, engine=engine)

    @staticmethod
    def backward(ctx, grad):
        x, qwt, swt = ctx.saved_tensors
        qg, sg = quantize_rows(grad, torch.float8_e5m2)
        dx = native_mm(qg, sg, qwt, swt, out_dtype=ctx.input_dtype, engine=ctx.engine)
        # The weight-gradient reduction is over tokens, so quantize that
        # orientation independently. Padding is included in measured work.
        pad = (-x.shape[-2]) % 16
        gt = F.pad(grad.mT, (0, pad)).contiguous()
        xt = F.pad(x.mT, (0, pad)).contiguous()
        qgt, sgt = quantize_rows(gt, torch.float8_e5m2)
        qxt, sxt = quantize_rows(xt)
        dw = native_mm(qgt, sgt, qxt, sxt, out_dtype=ctx.master_dtype, engine=ctx.engine)
        return dx, dw, None, None, None, None, None


def linear(x, master, packed=None, *, engine='quack'):
    if x.device.type != 'cuda' or master.device != x.device:
        raise ValueError('FP8 research compute requires tensors on one CUDA device')
    if x.shape[-1] != master.shape[-1] or any(d % 16 for d in master.shape[-2:]):
        raise ValueError('FP8 research matrices need compatible widths divisible by 16')
    if x.ndim != master.ndim or x.shape[:-2] != master.shape[:-2]:
        raise ValueError('FP8 research linear needs matching batch axes')
    if packed is None:
        packed = WeightCache().bind((master,))[0]
    return _Linear.apply(x, master, *packed, engine)


def compute(src, levels, experts, tokens, scores, *, cache=None, generation=None,
            recompute=False, **unused):
    if len(levels) != 1 or src.device.type != 'cuda':
        raise ValueError('FP8 research experts require CUDA and depth-1 experts')
    if not experts.numel():
        return torch.zeros_like(src)
    weights = levels[0]
    if any(w.dtype != torch.float32 for w in weights):
        raise ValueError('FP8 research experts bind original FP32 masters')
    n, width, hidden = weights[0].shape
    if hidden != src.shape[-1] or hidden % 16 or width % 16:
        raise ValueError('FP8 expert widths must be compatible and divisible by 16')
    cache = cache if cache is not None else WeightCache()
    packed = cache.bind(weights, generation)
    order = torch.argsort(experts, stable=True)
    experts, tokens, scores = experts[order], tokens[order], scores[order]
    counts = torch.bincount(experts, minlength=n)
    capacity = math.ceil(int(counts.max()) / 16) * 16
    starts = counts.cumsum(0) - counts
    slots = torch.arange(experts.numel(), device=src.device) - starts[experts]

    def run(x, w1, w3, w2, route_scores):
        # Rowwise scales look only across this token's channels. Empty/padded
        # rows stay zero and contribute no weight gradient.
        buf = torch.zeros(n, capacity, hidden, device=x.device, dtype=torch.bfloat16)
        buf[experts, slots] = x[tokens].to(torch.bfloat16)
        gate = linear(buf, w1, packed[0])
        up = linear(buf, w3, packed[1])
        y = linear(F.silu(gate) * up, w2, packed[2])[experts, slots].to(x.dtype)
        out = torch.zeros_like(x)
        out.index_add_(0, tokens, y * route_scores.unsqueeze(-1))
        return out

    args = (src, *weights, scores)
    # Quantized weights and route metadata bind BEFORE checkpointing. Replays
    # never read updated master values or a subsequently replaced cache.
    return checkpoint(run, *args, use_reentrant=False) if recompute else run(*args)


@contextmanager
def installed(model, *, use_cache=True):
    """Inject this research backend into one isolated experiment process."""
    import minagi.pool as pool_module
    pool = model.pool
    sites = [m for m in model.modules() if isinstance(m, pool_module.PooledMLP)]
    old = [(site, site.expert_backend, getattr(site, 'fast_inference', True)) for site in sites]
    cache = WeightCache()
    def dispatch(src, levels, experts, tokens, scores, **kwargs):
        if not use_cache:
            cache.clear()
        generation = (pool._stepped, pool.swaps, tuple(pool.slots))
        return compute(src, levels, experts, tokens, scores, cache=cache,
                       generation=generation, **kwargs)
    for site in sites:
        site.expert_backend, site.fast_inference = 'reference', False
    try:
        # Each custom autograd use returns its FP32 gradient to the master.
        # Cached detached FP8 weights provide immutable replay data.
        with patch.object(pool, 'training_weights', lambda dtype: (pool.w1,pool.w3,pool.w2)), \
             patch.object(pool_module, 'compute_experts', dispatch):
            yield cache
    finally:
        for site, backend, few in old:
            site.expert_backend, site.fast_inference = backend, few
