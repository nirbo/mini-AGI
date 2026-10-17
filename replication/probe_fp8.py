"""Native ordinary-FP8 layer qualification and complete training benchmarks.

Run under replication.gpu_monitor. No model/checkpoint files are opened.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from minagi.expert_compute import compute_experts
from minagi.precision import amp,set_compute_dtype
from minagi.paged import _CachedCast
from replication import fp8_experts as fp8
from replication.probe_nvfp4 import benchmark,error_metrics


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--seconds',type=float,default=3.)
    ap.add_argument('--phase',choices=('linear','experts'),required=True)
    ap.add_argument('--large',action='store_true')
    ap.add_argument('--reverse',action='store_true',help='Reverse shape/method order for the second timing repeat.')
    args=ap.parse_args()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.75)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(42)
    set_compute_dtype('bf16')
    report=dict(configuration=vars(args)|{'out':str(args.out)},cases=[],status='running',
                source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                               ('replication/fp8_experts.py',__file__)},
                limits=['Synthetic layer timings, not full-model throughput or learning quality.',
                        'Ordinary FP8 rowwise scales; E4M3 forward and E5M2 gradient signals.',
                        'FP32 masters/gradients. QuACK FP32 output plus separate rescaling.',
                        'FP8 expert dispatch is padded, unlike sparse Sonic BF16.'])
    if args.phase=='linear':
        shapes=[(128,512,2048),(1024,512,2048),(128,2048,512),(1024,2048,512)]
        if args.large:shapes=[(4096,4096,4096)]
        if args.reverse:shapes.reverse()
        for m,k,n in shapes:
            x=torch.randn(m,k,device='cuda',requires_grad=True)
            w=(torch.randn(n,k,device='cuda')/k**.5).requires_grad_()
            g=torch.randn(m,n,device='cuda',dtype=torch.bfloat16)
            ref_y=F.linear(x.bfloat16(),w.bfloat16())
            ref_y.backward(g)
            reference=(ref_y.detach().clone(),x.grad.detach().clone(),w.grad.detach().clone())
            del ref_y
            methods=['bf16','fp8_torch','fp8_quack']
            if args.reverse:methods.reverse()
            for method in methods:
                for cached in ((True,False) if args.reverse else (False,True)):
                    cache=fp8.WeightCache()
                    wb=w.detach().bfloat16() if method=='bf16' and cached else None
                    def step():
                        x.grad=w.grad=None
                        if method=='bf16':
                            y=F.linear(x.bfloat16(),_CachedCast.apply(w,wb) if cached else w.bfloat16())
                        else:
                            if not cached:cache.clear()
                            packed=cache.bind((w,))[0]
                            y=fp8.linear(x,w,packed,engine=method.removeprefix('fp8_'))
                        y.backward(g)
                        return y
                    case=dict(shape_mkn=[m,k,n],method=method,cached=cached)
                    y=step()
                    case['bf16_reference_errors']=[error_metrics(a,b) for a,b in zip((y,x.grad,w.grad),reference)]
                    if any(not e['finite'] or e['relative_l2']>.2 for e in case['bf16_reference_errors']):
                        raise AssertionError('linear FP8 approximation exceeds the declared error bound')
                    case['benchmark']=benchmark(step,m,args.seconds)
                    assert all(bool(torch.isfinite(t).all()) for t in (x.grad,w.grad))
                    case.update(status='passed',cache_conversions=cache.conversions,cache_hits=cache.hits)
                    report['cases'].append(case)
                    args.out.write_text(json.dumps(report,indent=2)+'\n')
                    print(json.dumps(case),flush=True)
            del x,w,g
            torch.cuda.empty_cache()
    else:
        tokens,experts,topk,hidden,width=(4096,32,8,512,2048)
        if args.large:tokens,experts,topk,hidden,width=(2048,4,2,4096,16384)
        x=torch.randn(tokens,hidden,device='cuda',requires_grad=True)
        ws=tuple((torch.randn(*s,device='cuda')/s[-1]**.5).requires_grad_() for s in
                 ((experts,width,hidden),(experts,width,hidden),(experts,hidden,width)))
        tt=torch.arange(tokens,device='cuda').repeat_interleave(topk)
        ee=(torch.arange(tokens*topk,device='cuda')%experts).long()
        scores=torch.full((len(tt),),1/topk,device='cuda',requires_grad=True)
        g=torch.randn_like(x)
        scenarios=[(m,c) for m in ('reference','sonic','fp8') for c in (False,True)]
        if args.reverse:scenarios.reverse()
        for method,cached in scenarios:
            cache=fp8.WeightCache()
            wb=tuple(w.detach().bfloat16() for w in ws) if method!='fp8' and cached else None
            def step():
                for p in (x,*ws,scores):p.grad=None
                with amp('cuda'):
                    if method=='fp8':
                        if not cached:cache.clear()
                        y=fp8.compute(x,[ws],ee,tt,scores,cache=cache)
                    else:
                        live=tuple(_CachedCast.apply(w,b) for w,b in zip(ws,wb)) if cached else ws
                        y=compute_experts(x,[live],ee,tt,scores,backend=method,padded=method=='reference')
                y.backward(g)
            case=dict(geometry=dict(tokens=tokens,experts=experts,topk=topk,hidden=hidden,width=width),
                      method=method,cached=cached)
            case['benchmark']=benchmark(step,tokens,args.seconds)
            assert all(bool(torch.isfinite(p.grad).all()) for p in (x,*ws,scores))
            case.update(status='passed',cache_conversions=cache.conversions,cache_hits=cache.hits)
            report['cases'].append(case)
            args.out.write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(case),flush=True)
    report['status']='passed'
    args.out.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
