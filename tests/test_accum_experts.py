"""Qualify research accumulator changes under explicit GPU power monitoring."""

import os
import unittest
from functools import partial
from unittest.mock import patch

import torch
import torch.nn.functional as F

from replication import accum_experts as accum
from replication import fp8_experts as route
from tests import test_fp8_experts as fp8_tests

relative = fp8_tests.relative


class AccumulatorGuards(unittest.TestCase):
    def test_invalid_accumulator_and_cpu_linear(self):
        with self.assertRaisesRegex(ValueError,'fp32 or fp16'):
            with accum.kernels('bf16'):pass
        with self.assertRaisesRegex(ValueError,'CUDA'):
            accum.linear(torch.ones(2,16),torch.ones(16,16))

    def test_cache_owns_immutable_data_and_invalidates(self):
        w=torch.ones(16,32)
        cache=accum.CastCache(torch.float16)
        data=cache.bind((w,),0)
        self.assertIs(cache.bind((w,),0),data)
        w.add_(1);cache.bind((w,),0)
        w.data=w.data.clone();cache.bind((w,),0)
        w.data.add_(1);cache.bind((w,),1)
        cache.bind((w.clone(),),1)
        self.assertEqual(cache.conversions,5)
        self.assertEqual(cache.hits,1)
        torch.testing.assert_close(data[0],torch.ones_like(data[0]),rtol=0,atol=0)
        self.assertIsNotNone(cache.owners)


@unittest.skipUnless(os.environ.get('MINAGI_ACCUM_TEST_CUDA')=='1','explicit monitored accumulator qualification')
class AccumulatorCudaTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)
        torch.manual_seed(42)
        old=torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        self.addCleanup(torch.use_deterministic_algorithms,old)
        old=torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32=False
        self.addCleanup(setattr,torch.backends.cuda.matmul,'allow_tf32',old)

    def test_native_forward_dgrad_wgrad_against_float64_decoded_operands(self):
        for recipe,(dtype,accumulator,split) in accum.RECIPES.items():
            for batched in (False,True):
                with self.subTest(recipe=recipe,batched=batched),accum.kernels(accumulator):
                    prefix=(2,) if batched else ()
                    x=torch.randn(*prefix,32,64,device='cuda',requires_grad=True)
                    w=(torch.randn(*prefix,128,64,device='cuda')*.1).requires_grad_()
                    original=accum.mm;seen=[]
                    def checked(a,b,**kw):
                        actual=original(a,b,**kw)
                        oracle=a.double()@b.double()
                        self.assertLess(relative(actual,oracle),.005)
                        seen.append((a.dtype,b.dtype,actual.dtype))
                        return actual
                    with patch.object(accum,'mm',checked):
                        y=accum.linear(x,w,dtype=dtype,split_k=split)
                        y.backward(torch.randn_like(y))
                    self.assertEqual(len(seen),3)
                    self.assertTrue(all(a==b==dtype for a,b,_ in seen))
                    self.assertEqual(w.grad.dtype,torch.float32)
                    self.assertEqual(x.grad.dtype,torch.float32)

    def test_shorter_fp16_partials_improve_long_reductions(self):
        a=torch.randn(128,4096,device='cuda',dtype=torch.float16)
        b=torch.randn(128,4096,device='cuda',dtype=torch.float16).T
        ref=a.double()@b.double()
        with accum.kernels('fp16'):
            full=accum.mm(a,b)
            split=accum.mm(a,b,split_k=4)
            replay=accum.mm(a,b,split_k=4)
        self.assertLess(relative(split,ref),relative(full,ref))
        torch.testing.assert_close(split,replay,rtol=0,atol=0)

    def test_fp16_accumulator_overflow_is_detectable_and_split_can_avoid_it(self):
        # Final sum is finite; full FP16 partial sums exceed 65504.
        a=torch.ones(128,4096,device='cuda',dtype=torch.float16)
        b=torch.full((128,4096),40.,device='cuda',dtype=torch.float16)
        b[:,2048:]=-40.
        with accum.kernels('fp32'):high=accum.mm(a,b.T)
        with accum.kernels('fp16'):
            low=accum.mm(a,b.T)
            split=accum.mm(a,b.T,split_k=4)
        self.assertTrue(bool(torch.isfinite(high).all()))
        self.assertFalse(bool(torch.isfinite(low).all()))
        self.assertTrue(bool(torch.isfinite(split).all()))
        self.assertLess(float(split.abs().max()),1.)

    def test_all_mixture_derivatives_replay_and_cache_generations(self):
        for recipe,(dtype,accumulator,split) in accum.RECIPES.items():
            with self.subTest(recipe=recipe),accum.kernels(accumulator), \
                 patch.object(route,'linear',partial(accum.linear,dtype=dtype,split_k=split)):
                fixture=fp8_tests.FP8CudaTests('runTest')
                x,ws,e,t,s=fixture.inputs()
                params=[x,*ws,s]
                ref=torch.zeros_like(x)
                for expert,token,score in zip(e.tolist(),t.tolist(),s):
                    xx=x[token:token+1].to(dtype).float()
                    a,b,c=(w[expert].to(dtype).float() for w in ws)
                    gate=(xx@a.T).bfloat16();up=(xx@b.T).bfloat16()
                    active=(F.silu(gate)*up).to(dtype).float()
                    y=(active@c.T).bfloat16().float()
                    ref=ref.index_add(0,t.new_tensor([token]),y*score)
                upstream=torch.randn_like(x)
                expected=torch.autograd.grad(ref,params,upstream)
                cache=accum.CastCache(dtype)
                actual=route.compute(x,[ws],e,t,s,cache=cache,generation=0)
                grads=torch.autograd.grad(actual,params,upstream)
                self.assertLess(relative(actual,ref),.03)
                for g,r in zip(grads,expected):self.assertLess(relative(g,r),.04)
                y=route.compute(x,[ws],e,t,s,cache=cache,generation=0,recompute=True)
                for w in ws:w.data.mul_(2)
                cache.bind(ws,generation=1)
                replay=torch.autograd.grad(y,params,upstream)
                torch.testing.assert_close(y,actual,rtol=0,atol=0)
                for a,b in zip(replay,grads):torch.testing.assert_close(a,b,rtol=0,atol=0)
                self.assertGreater(abs(float(grads[-1][0])),0)
                self.assertEqual(int(actual[1].count_nonzero()),0)
                for g in grads[1:4]:self.assertEqual(int(g[3].count_nonzero()),0)

    def test_inference_prepared_cache_and_kernel_context_restore(self):
        import importlib
        module=importlib.import_module('quack.gemm')
        original_compile=module._compile_gemm
        original_plan=module._gemm_plan_cache
        w=torch.randn(32,64,device='cuda',requires_grad=True)
        cache=accum.CastCache(torch.float16)
        with torch.inference_mode():packed=cache.bind((w,))[0]
        with accum.kernels('fp16'):
            x=torch.randn(32,64,device='cuda',requires_grad=True)
            accum.linear(x,w,packed).square().sum().backward()
        self.assertTrue(bool(torch.isfinite(w.grad).all()))
        self.assertIs(module._compile_gemm,original_compile)
        self.assertIs(module._gemm_plan_cache,original_plan)

    def test_model_causal_prefix_and_checkpoint_cache_equivalence(self):
        from tests.test_causality import CausalityTests
        from minagi.precision import amp,compute_dtype,set_compute_dtype
        from minagi.pool import PooledMLP
        old=compute_dtype();set_compute_dtype('bf16');self.addCleanup(set_compute_dtype,old)
        for recipe in accum.RECIPES:
            reference=None
            for cached,replay in ((False,False),(True,False),(True,True)):
                fixture=CausalityTests('runTest');self.addCleanup(fixture.doCleanups)
                model=fixture.model(device='cuda',steps=3,recur=2,coda=1,freeze=False).train()
                for site in model.modules():
                    if isinstance(site,PooledMLP):site.grad_checkpoint=replay
                text=torch.arange(24,device='cuda')[None]
                with accum.installed(model,recipe=recipe,use_cache=cached),amp('cuda'):
                    out,loss=model(text,torch.zeros_like(text));loss.backward()
                grads={n:p.grad.detach().clone() for n,p in model.named_parameters() if p.grad is not None}
                if reference:
                    torch.testing.assert_close(out,reference[0],rtol=0,atol=0)
                    for name,g in grads.items():torch.testing.assert_close(g,reference[1][name],rtol=0,atol=0,msg=name)
                else:reference=(out.detach(),grads)
            for training,capacity in ((False,0),(True,1.5)):
                fixture=CausalityTests('runTest');self.addCleanup(fixture.doCleanups)
                model=fixture.model(device='cuda',resident=8,capacity=capacity,steps=3,temperature=.7)
                def head(x):
                    return torch.cat([F.linear(row[None],model.head.weight) for row in x.flatten(0,1)]).view(*x.shape[:2],-1)
                model.head.forward=head
                with accum.installed(model,recipe=recipe),amp('cuda'):
                    fixture.assert_prefix(model,training=training,batch=2)


if __name__=='__main__':unittest.main()
