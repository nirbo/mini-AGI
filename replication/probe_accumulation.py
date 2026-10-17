"""Controlled native SM120 accumulator probes; run externally power monitored."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from minagi.expert_compute import compute_experts
from minagi.paged import _CachedCast
from minagi.precision import amp, set_compute_dtype
from replication import accum_experts as accum
from replication import fp8_experts as route
from replication.probe_nvfp4 import benchmark, error_metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase', choices=('gemm', 'linear', 'experts'), required=True)
    ap.add_argument('--large', action='store_true')
    ap.add_argument('--reverse', action='store_true')
    ap.add_argument('--seconds', type=float, default=3.)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.seconds <= 0:
        ap.error('--seconds must be positive')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.75)
    set_compute_dtype('bf16')
    report = dict(status='running', cases=[], configuration=vars(args) | {'out': str(args.out)},
                  source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                 (Path(__file__), Path(accum.__file__), Path(route.__file__))},
                  limits=['Synthetic fixtures; no optimizer or full-model fit/convergence qualification.',
                          'All timings include Python launch and synchronization overhead.',
                          'Sonic is fused/sparse; the controlled research paths share padded dispatch.',
                          'Cached synthetic weights do not change during the timing loop.'])
    methods = list(accum.RECIPES)
    if args.phase == 'experts':
        methods = ['sonic', *methods]
    if args.reverse:
        methods.reverse()
    def save(case):
        case['status'] = 'passed'
        report['cases'].append(case)
        args.out.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(case), flush=True)
    if args.phase in ('gemm', 'linear'):
        shapes = [(128,512,2048), (128,2048,512)]
        if args.phase == 'gemm':
            shapes += [(2048,128,512)]
        if args.large:
            shapes = [(4096,4096,4096)]
        for m,k,n in shapes:
            # Shape-based seeding keeps inputs identical across reversed repeats.
            torch.manual_seed(42 + m + k + n)
            x = torch.randn(m,k,device='cuda',requires_grad=True)
            w = (torch.randn(n,k,device='cuda') / k**.5).requires_grad_()
            g = torch.randn(m,n,device='cuda',dtype=torch.bfloat16)
            if args.phase == 'linear':
                yr = F.linear(x.bfloat16(),w.bfloat16())
                yr.backward(g)
                references = (yr.detach().clone(),x.grad.detach().clone(),w.grad.detach().clone())
                del yr
            for recipe in methods:
                dtype, accumulator, split = accum.RECIPES[recipe]
                with accum.kernels(accumulator):
                    cache = accum.CastCache(dtype)
                    wb = cache.bind((w,))[0]
                    if args.phase == 'gemm':
                        xb = x.detach().to(dtype)
                        ref = xb.double() @ wb.double().T
                        def step():
                            return accum.mm(xb, wb.T, split_k=split)
                        y = step()
                        errors = error_metrics(y,ref)
                        # This qualification bounds normal-distribution error;
                        # adversarial range/cancellation cases live in tests.
                        assert errors['finite'] and errors['relative_l2'] < .02, errors
                        del ref
                        case = dict(shape_mkn=[m,k,n], recipe=recipe, decoded_operand_error=errors)
                    else:
                        def step():
                            x.grad = w.grad = None
                            y = accum.linear(x,w,wb,split_k=split)
                            y.backward(g)
                            return y
                        y = step()
                        errors = [error_metrics(a,b) for a,b in zip((y,x.grad,w.grad),references)]
                        assert all(e['finite'] and e['relative_l2'] < .05 for e in errors), errors
                        case = dict(shape_mkn=[m,k,n],recipe=recipe,bf16_reference_errors=errors)
                    case['benchmark'] = benchmark(step,m,args.seconds)
                    save(case)
            del x,w,g,cache,wb,y
            torch.cuda.empty_cache()
    else:
        tokens,experts,topk,hidden,width = (4096,32,8,512,2048)
        if args.large:
            tokens,experts,topk,hidden,width = (2048,4,2,4096,16384)
        torch.manual_seed(42)
        x = torch.randn(tokens,hidden,device='cuda',requires_grad=True)
        ws = tuple((torch.randn(*s,device='cuda') / s[-1]**.5).requires_grad_() for s in
                   ((experts,width,hidden),(experts,width,hidden),(experts,hidden,width)))
        tt = torch.arange(tokens,device='cuda').repeat_interleave(topk)
        ee = torch.arange(tokens*topk,device='cuda') % experts
        scores = torch.full((len(tt),),1/topk,device='cuda',requires_grad=True)
        grad = torch.randn_like(x)
        for recipe in methods:
            dtype, accumulator, split = accum.RECIPES.get(recipe,(torch.bfloat16,'fp32',1))
            with accum.kernels(accumulator):
                cache = accum.CastCache(dtype)
                packed = cache.bind(ws)
                def step():
                    for p in (x,*ws,scores):p.grad = None
                    with amp('cuda'):
                        if recipe == 'sonic':
                            live = tuple(_CachedCast.apply(w,b) for w,b in zip(ws,packed))
                            y = compute_experts(x,[live],ee,tt,scores,backend='sonic')
                        else:
                            from functools import partial
                            from unittest.mock import patch
                            with patch.object(route,'linear',partial(accum.linear,dtype=dtype,split_k=split)):
                                y = route.compute(x,[ws],ee,tt,scores,cache=cache)
                    y.backward(grad)
                case = dict(recipe=recipe,geometry=dict(tokens=tokens,experts=experts,topk=topk,
                                                       hidden=hidden,width=width))
                case['benchmark'] = benchmark(step,tokens,args.seconds)
                assert all(bool(torch.isfinite(p.grad).all()) for p in (x,*ws,scores))
                save(case)
    report['status'] = 'passed'
    args.out.write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
