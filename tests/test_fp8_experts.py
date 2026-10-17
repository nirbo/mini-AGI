"""Ordinary FP8 research qualification; opt into CUDA under GPU monitoring."""

import os
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from replication import fp8_experts as fp8


def relative(a, b):
    return float((a.detach().float()-b.detach().float()).norm()/b.detach().float().norm().clamp_min(1e-30))


class FP8Guards(unittest.TestCase):
    def test_cpu_compute_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'CUDA'):
            fp8.linear(torch.ones(2,16),torch.ones(16,16))

    def test_cache_identity_versions_storage_and_generation(self):
        calls=[]
        def quant(x):
            calls.append(x)
            return x.clone(),torch.ones(*x.shape[:-1],1)
        weights=[torch.ones(16,32)]
        cache=fp8.WeightCache()
        with patch.object(fp8,'quantize_rows',quant):
            original=cache.bind(weights,0)
            self.assertIs(cache.bind(weights,0),original)
            self.assertEqual(len(calls),2)
            weights[0].add_(1)
            self.assertIsNot(cache.bind(weights,0),original)
            weights[0].data=weights[0].data.clone()
            cache.bind(weights,0)
            weights[0].data.add_(1)  # pager writes need its explicit generation
            cache.bind(weights,1)
            cache.bind([weights[0].clone()],1)
        self.assertEqual(cache.conversions,5)
        self.assertEqual(cache.hits,1)
        self.assertTrue(torch.equal(original[0][0],torch.ones(16,32)))


@unittest.skipUnless(os.environ.get('MINAGI_FP8_TEST_CUDA')=='1','explicit monitored FP8 CUDA qualification')
class FP8CudaTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)
        old_limit=torch._dynamo.config.recompile_limit
        torch._dynamo.config.recompile_limit=64
        self.addCleanup(setattr,torch._dynamo.config,'recompile_limit',old_limit)
        torch.manual_seed(42)
        old=torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        self.addCleanup(torch.use_deterministic_algorithms,old)
        old=torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32=False
        self.addCleanup(setattr,torch.backends.cuda.matmul,'allow_tf32',old)

    def test_three_native_gemms_against_decoded_operands(self):
        for engine in ('torch','quack'):
            with self.subTest(engine=engine):
                x=torch.randn(17,64,device='cuda',requires_grad=True)
                w=(torch.randn(128,64,device='cuda')*.1).requires_grad_()
                records=[]
                original=fp8.native_mm
                def record(a,sa,b,sb,**kwargs):
                    y=original(a,sa,b,sb,**kwargs)
                    expected=(a.float()*sa)@(b.float()*sb).mT
                    self.assertLess(relative(y,expected),.005)
                    records.append((a.dtype,b.dtype,y.dtype))
                    return y
                with patch.object(fp8,'native_mm',record):
                    y=fp8.linear(x,w,engine=engine)
                    y.backward(torch.randn_like(y))
                self.assertEqual(len(records),3)
                self.assertEqual(records[0][:2],(torch.float8_e4m3fn,torch.float8_e4m3fn))
                self.assertEqual(records[1][:2],(torch.float8_e5m2,torch.float8_e4m3fn))
                self.assertEqual(records[2],(torch.float8_e5m2,torch.float8_e4m3fn,torch.float32))
                self.assertEqual(x.grad.dtype,torch.float32)
                self.assertEqual(w.grad.dtype,torch.float32)

    def inputs(self):
        x=torch.randn(19,32,device='cuda',requires_grad=True)
        ws=tuple((torch.randn(*s,device='cuda')*.1).requires_grad_()
                 for s in ((4,64,32),(4,64,32),(4,32,64)))
        tokens=torch.tensor([0,0,2,4,4,6,7,8,9,12,16,18],device='cuda')
        experts=torch.tensor([0,1,0,1,2,0,2,0,2,1,0,2],device='cuda')
        scores=torch.linspace(-.3,.5,len(tokens),device='cuda').requires_grad_()
        with torch.no_grad():scores[0]=0
        return x,ws,experts,tokens,scores

    def test_all_mixture_derivatives_and_checkpoint_replay(self):
        x,ws,e,t,s=self.inputs()
        params=[x,*ws,s]
        reference=torch.zeros_like(x)
        for expert,token,score in zip(e.tolist(),t.tolist(),s):
            xx=x[token:token+1].bfloat16()
            a,b,c=(w[expert].bfloat16() for w in ws)
            y=(F.silu(xx@a.T)*(xx@b.T))@c.T
            reference=reference.index_add(0,t.new_tensor([token]),y.float()*score)
        upstream=torch.randn_like(x)
        expected=torch.autograd.grad(reference,params,upstream)
        previous=None
        for replay in (False,True):
            actual=fp8.compute(x,[ws],e,t,s,recompute=replay)
            grads=torch.autograd.grad(actual,params,upstream)
            self.assertLess(relative(actual,reference),.12)
            for g,r in zip(grads,expected):
                self.assertTrue(bool(torch.isfinite(g).all()))
                self.assertLess(relative(g,r),.18)
            self.assertGreater(abs(float(grads[-1][0])),0)
            self.assertEqual(int(actual[1].count_nonzero()),0)
            for g in grads[1:4]:self.assertEqual(int(g[3].count_nonzero()),0)
            if previous:
                torch.testing.assert_close(actual,previous[0],rtol=0,atol=0)
                for g,r in zip(grads,previous[1]):torch.testing.assert_close(g,r,rtol=0,atol=0)
            previous=(actual,grads)

    def test_future_outliers_do_not_change_prefix(self):
        x,ws,e,t,s=self.inputs()
        expected=fp8.compute(x,[ws],e,t,s)
        future=torch.full((33,32),100.,device='cuda')
        tt=torch.arange(19,52,device='cuda').repeat_interleave(2)
        ee=torch.tensor([0,2]*33,device='cuda')
        actual=fp8.compute(torch.cat((x,future)),[ws],torch.cat((e,ee)),torch.cat((t,tt)),
                           torch.cat((s,torch.full((66,),.5,device='cuda'))))
        torch.testing.assert_close(actual[:19],expected,rtol=0,atol=0)

    def test_replay_keeps_bound_quantized_weights_after_unversioned_paging(self):
        x,ws,e,t,s=self.inputs()
        upstream=torch.randn_like(x)
        baseline=fp8.compute(x,[ws],e,t,s,recompute=False)
        expected=torch.autograd.grad(baseline,[x,*ws,s],upstream)
        cache=fp8.WeightCache()
        y=fp8.compute(x,[ws],e,t,s,recompute=True,cache=cache,generation=0)
        for w in ws:w.data.mul_(2)
        cache.bind(ws,generation=1)
        actual=torch.autograd.grad(y,[x,*ws,s],upstream)
        torch.testing.assert_close(y,baseline,rtol=0,atol=0)
        for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_model_cache_checkpoint_and_paging_agree(self):
        from tests.test_causality import CausalityTests
        from minagi.pool import PooledMLP
        from minagi.precision import amp,compute_dtype,set_compute_dtype
        old=compute_dtype();set_compute_dtype('bf16');self.addCleanup(set_compute_dtype,old)
        reference=None
        for cached,replay in ((False,False),(True,False),(True,True)):
            model=CausalityTests.model(self,device='cuda',steps=3,recur=2,coda=1,freeze=False).train()
            for site in model.modules():
                if isinstance(site,PooledMLP):site.grad_checkpoint=replay
            text=torch.arange(24,device='cuda')[None]
            losses=[];outputs=[];caches=model.empty_caches()
            with fp8.installed(model,use_cache=cached) as cache:
                for start,stop in ((0,1),(1,12),(12,24)):
                    with amp('cuda'):
                        out,loss=model(text[:,start:stop],torch.zeros_like(text[:,start:stop]),
                                       caches=caches,pos_offset=start)
                    outputs.append(out.detach());losses.append(loss*(stop-start)/24)
                sum(losses).backward()
                grads={n:p.grad.detach().clone() for n,p in model.named_parameters() if p.grad is not None}
                self.assertTrue(all(bool(torch.isfinite(g).all()) for g in grads.values()))
                if cached:self.assertGreater(cache.hits,0)
                result=(torch.cat(outputs,1),grads)
                if reference:
                    torch.testing.assert_close(result[0],reference[0],rtol=0,atol=0)
                    for n,g in grads.items():torch.testing.assert_close(g,reference[1][n],rtol=0,atol=0,msg=n)
                else:reference=result

    def test_model_causal_prefix_with_capacity_and_stochastic_routing(self):
        from tests.test_causality import CausalityTests
        from minagi.precision import amp,compute_dtype,set_compute_dtype
        old=compute_dtype();set_compute_dtype('bf16');self.addCleanup(set_compute_dtype,old)
        for training,capacity in ((False,0),(True,0),(False,1.5),(True,1.5)):
            fixture=CausalityTests('runTest')
            self.addCleanup(fixture.doCleanups)
            model=fixture.model(device='cuda',resident=8,capacity=capacity,steps=3,temperature=.7)
            def head(x):
                return torch.cat([F.linear(row[None],model.head.weight) for row in x.flatten(0,1)]).view(*x.shape[:2],-1)
            model.head.forward=head
            with self.subTest(training=training,capacity=capacity),fp8.installed(model),amp('cuda'):
                fixture.assert_prefix(model,training=training,batch=2)

    def test_gefen_updates_and_page_changes_invalidate_quantized_cache(self):
        from tests.test_causality import CausalityTests
        from minagi.optimizers import make_optimizer
        model=CausalityTests.model(self,device='cuda',resident=2,experts=4)
        pool=model.pool
        opt=make_optimizer(model.parameters(),pool=pool,optimizer='gefen',lr=3e-5,
                           betas=(.9,.95),fused=True)
        pool.attach_optimiser(opt)
        cache=fp8.WeightCache()
        for selected in ((0,1),(2,3),(0,1)):
            pool.begin_text()
            pool.admit(torch.tensor([[selected]],device='cuda'))
            x=torch.randn(7,32,device='cuda',requires_grad=True)
            t=torch.arange(7,device='cuda').repeat_interleave(2)
            e=torch.tensor([0,1]*7,device='cuda')
            s=torch.full((14,),.5,device='cuda')
            weights=(pool.w1,pool.w3,pool.w2)
            generation=(pool._stepped,pool.swaps,tuple(pool.slots))
            y=fp8.compute(x,[weights],e,t,s,cache=cache,generation=generation)
            fresh=fp8.compute(x,[weights],e,t,s)
            torch.testing.assert_close(y,fresh,rtol=0,atol=0)
            opt.zero_grad(set_to_none=True)
            y.square().sum().backward()
            for w in weights:self.assertEqual(w.grad.dtype,torch.float32)
            before=[w.detach().clone() for w in weights]
            opt.step()
            self.assertTrue(any(not torch.equal(w,old) for w,old in zip(weights,before)))
            self.assertEqual(pool.tiers.writebacks,0)
        self.assertEqual(cache.conversions,3)


if __name__=='__main__':
    unittest.main()
