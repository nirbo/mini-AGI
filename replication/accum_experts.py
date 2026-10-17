"""Research-only accumulator controls over existing QuACK SM120 GEMMs.

No installed package is edited. FP16 accumulation uses the existing kernel's
acc_dtype constructor, bypassing the public helper's fixed Float32 argument.
Compilation and dispatch caches are separated from the normal FP32 path.
"""

from contextlib import contextmanager
import importlib
from functools import partial
from pathlib import Path
from unittest.mock import patch

import torch


@contextmanager
def kernels(accumulator='fp32'):
    if accumulator not in ('fp32', 'fp16'):
        raise ValueError('accumulator must be fp32 or fp16')
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        raise ValueError('accumulator research kernels require an SM120-family CUDA GPU')
    import cutlass
    import quack.cache as cache
    from quack.cache import jit_cache
    module = importlib.import_module('quack.gemm')
    interface = importlib.import_module('quack.gemm_interface')
    original_compile = module._compile_gemm
    original_class = module.GemmDefaultSm120

    class FP16Accumulator(original_class):
        def __init__(self, acc_dtype, a_dtype, *args, **kwargs):
            if a_dtype != cutlass.Float16:
                raise ValueError('FP16 accumulation experiment requires FP16 operands')
            super().__init__(cutlass.Float16, a_dtype, *args, **kwargs)

    # A distinct qualified name and an isolated cache root prevent an existing
    # FP32 object from being mistaken for an FP16-accumulating kernel.
    @jit_cache
    def compile_fp16(*args, **kwargs):
        from quack import sm80_utils
        # The shared partition helper also hardcodes FP32 register storage.
        # Its fragment layout comes from the selected MMA atom; only the
        # allocated fragment element type changes here.
        with patch.object(module, 'GemmDefaultSm120', FP16Accumulator), \
             patch.object(sm80_utils, 'Float32', cutlass.Float16):
            return original_compile.__wrapped__(*args, **kwargs)

    root = Path('runs/accum-quack-cache').resolve()
    root.mkdir(parents=True, exist_ok=True)
    with patch.object(cache, 'CACHE_DIR', str(root)), \
         patch.object(cache, 'EXTRA_SOURCE_DIRS', cache.EXTRA_SOURCE_DIRS + [Path(__file__).parent]), \
         patch.object(module, '_compile_gemm', compile_fp16 if accumulator == 'fp16' else original_compile), \
         patch.object(module, '_gemm_plan_cache', {}), \
         patch.object(interface, '_gemm_iface_plan_cache', {}):
        yield


def mm(a, b, *, out_dtype=torch.float32, split_k=1):
    """Use QuACK's unmodified SM120 scheduling/mainloop/epilogue."""
    from quack.gemm_interface import gemm, SplitKMode
    return gemm(a, b, out_dtype=out_dtype, tuned=False,
                split_k=split_k, split_k_mode=SplitKMode.SEPARATE)


RECIPES = {
    'bf16-fp32': (torch.bfloat16, 'fp32', 1),
    'fp16-fp32': (torch.float16, 'fp32', 1),
    'fp16-fp16': (torch.float16, 'fp16', 1),
    'fp16-split4': (torch.float16, 'fp16', 4),
}


class CastCache:
    """Immutable compute weights, invalidated by owner/version/generation."""

    def __init__(self, dtype):
        self.dtype = dtype
        self.key = self.weights = self.owners = None
        self.conversions = self.hits = 0

    def clear(self):
        self.key = self.weights = self.owners = None

    @torch.no_grad()
    def bind(self, weights, generation=None):
        key = (generation, tuple((id(w), w.data_ptr(), w._version, w.device,
                                  w.dtype, tuple(w.shape), w.stride()) for w in weights))
        if key != self.key:
            with torch.inference_mode(False):
                self.weights = tuple(w.detach().to(self.dtype, copy=True) for w in weights)
            self.key, self.owners = key, tuple(weights)
            self.conversions += 1
        else:
            self.hits += 1
        return self.weights


class _Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, master, cast_weight, split_k):
        cast_input = x.to(cast_weight.dtype)
        ctx.save_for_backward(cast_input, cast_weight)
        ctx.input_dtype, ctx.master_dtype, ctx.split_k = x.dtype, master.dtype, split_k
        return mm(cast_input, cast_weight.mT, out_dtype=torch.bfloat16, split_k=split_k)

    @staticmethod
    def backward(ctx, grad):
        x, weight = ctx.saved_tensors
        g = grad.to(weight.dtype)
        dx = mm(g, weight, out_dtype=ctx.input_dtype, split_k=ctx.split_k)
        dw = mm(g.mT.contiguous(), x, out_dtype=ctx.master_dtype, split_k=ctx.split_k)
        return dx, dw, None, None


def linear(x, master, packed=None, *, dtype=torch.float16, split_k=1):
    if x.device.type != 'cuda' or master.device != x.device:
        raise ValueError('accumulator experiment requires tensors on one CUDA device')
    if x.shape[-1] != master.shape[-1] or x.ndim != master.ndim or x.shape[:-2] != master.shape[:-2]:
        raise ValueError('incompatible linear geometry')
    if packed is None:
        packed = CastCache(dtype).bind((master,))[0]
    return _Linear.apply(x, master, packed, split_k)


@contextmanager
def installed(model, *, recipe, use_cache=True):
    """Reuse qualified routing/checkpoint/cache generation hooks; no FP8 math."""
    from replication import fp8_experts as route
    dtype, accumulator, split_k = RECIPES[recipe]
    with kernels(accumulator), \
         patch.object(route, 'WeightCache', partial(CastCache, dtype)), \
         patch.object(route, 'linear', partial(linear, dtype=dtype, split_k=split_k)), \
         route.installed(model, use_cache=use_cache) as cache:
        yield cache
