"""Matched read-only checkpoint adaptation with controlled GEMM accumulators."""

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import types

os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'

import numpy as np
import torch

from minagi.ingest import as_stream
from minagi.optimizers import make_optimizer
from minagi.pool import PooledMLP
from minagi.precision import amp,set_compute_dtype
from minagi.recur import load_any
from replication import accum_experts as accum
from replication.compare_gefen_training import evaluate,weights_only
from train import _split_trunk_pool


def fingerprint(tree):
    digest=hashlib.sha256()
    def visit(v):
        if torch.is_tensor(v):
            digest.update(str((str(v.dtype),tuple(v.shape))).encode())
            digest.update(v.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(v,dict):
            for key in sorted(v,key=repr):digest.update(repr(key).encode());visit(v[key])
        elif isinstance(v,(list,tuple)):
            for item in v:visit(item)
        else:digest.update(repr(v).encode())
    visit(tree)
    return digest.hexdigest()


def pin_sonic_reduction():
    from sonicmoe.functional.reduction_over_k_gather import token_gather_sum_kernel
    from triton import Config
    token_gather_sum_kernel.configs=[Config({'BLOCK_H':256,'BLOCK_K':32},num_warps=4,num_stages=4)]
    token_gather_sum_kernel.cache.clear()


def source_hashes():
    paths=[Path('train.py'),Path('config.yaml'),Path(__file__),Path(accum.__file__),Path("replication/fp8_experts.py"),
           Path('replication/compare_gefen_training.py')]
    paths+=sorted(Path('minagi').glob('*.py'))
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--backend',choices=('sonic',*accum.RECIPES),required=True)
    ap.add_argument('--weights',type=Path,default=Path('/mnt/fast/models/mini-AGI/undertrained/weights'))
    ap.add_argument('--text',type=Path,default=Path('/mnt/fast/models/mini-AGI/evaluation-data/TinyStoriesV2-GPT4-valid.txt'))
    ap.add_argument('--steps',type=int,default=256)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--lr',type=float,default=3e-5)
    ap.add_argument('--aux-lr',type=float,default=1e-4)
    ap.add_argument('--context',type=int,default=512)
    ap.add_argument('--depth',type=int,default=6)
    ap.add_argument('--fingerprint',action='store_true')
    ap.add_argument('--no-cache',action='store_true')
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if not (args.context>0 and 0<args.steps<=262143//args.context and args.depth>0
            and args.context*16<262144):
        ap.error('invalid dimensions')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    report=dict(configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                source_sha256=source_hashes(),optimizer='gefen',trace=[],
                limits=['Short adaptation of one checkpoint at depth 6; not pretraining or a canonical document split.',
                        'Same fixed rates across precisions; no precision-specific hyperparameter tuning.',
                        'All touched experts cached in host RAM; checkpoint is read-only.',
                        'Research paths share padded QuACK GEMMs; Sonic uses fused sparse BF16.',
                        'SwiGLU/activation outputs remain BF16; masters and outer gradient buffers remain FP32.',
                        'FP16 operand range/rounding and FP16 internal accumulation are separate numerical changes.'])
    try:
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(4)
        torch.cuda.set_per_process_memory_fraction(.75)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.manual_seed(args.seed)
        set_compute_dtype('bf16')
        model,_=load_any(str(args.weights),torch.device('cuda'),read_only=True)
        model.cfg.max_steps=model.cfg.min_steps=model.cfg.bptt_window=args.depth
        model.cfg.train_steps_mean=0
        pool=model.pool
        if model.cfg.d_model!=512 or pool.resident!=32:
            raise ValueError('this screen is qualified for hidden 512 and 32 resident experts')
        if pool.tiers.ram or any(e>=0 for e in pool.slots):raise AssertionError('fresh empty pool required')
        pool.tiers.ram_capacity=pool.n_experts()
        pool.tiers._from_disk=types.MethodType(weights_only,pool.tiers)
        for site in model.modules():
            if isinstance(site,PooledMLP):site.expert_backend='sonic';site.grad_checkpoint=True
        pin_sonic_reduction()
        data=as_stream(str(args.text))
        if len(data)<524288+args.context*16+1:raise ValueError('text too short for held-out partitions')
        report['manifest_sha256']=hashlib.sha256((args.weights/'manifest.json').read_bytes()).hexdigest()
        report['text_sha256']=hashlib.sha256(args.text.read_bytes()).hexdigest()
        report['initial_model_sha256']=fingerprint(dict(model.state_dict()))
        indices=np.random.default_rng(args.seed).permutation(262143//args.context)[:args.steps]
        report['training_window_indices']=indices.tolist()
        mode=accum.installed(model,recipe=args.backend,use_cache=not args.no_cache) if args.backend!='sonic' else nullcontext(None)
        with mode as cache:
            report['initial_tune']=evaluate(model,data,262144,args.context,16)
            report['initial_test']=evaluate(model,data,524288,args.context,16)
            trunk,experts=_split_trunk_pool(model)
            slabs=[pool.w1,pool.w3,pool.w2]
            auxiliary=[p for p in experts if all(p is not w for w in slabs)]
            groups=[{'params':trunk,'lr':args.aux_lr*.2,'name':'trunk'},
                    {'params':auxiliary,'lr':args.aux_lr,'name':'pool_aux'},
                    {'params':slabs,'lr':args.lr,'name':'pool'}]
            opt=make_optimizer(groups,pool=pool,optimizer='gefen',lr=args.lr,
                               betas=(.9,.95),weight_decay=.1,fused=True)
            pool.attach_optimiser(opt)
            opt._gefen._gefen_codebook=None
            opt._gefen._gefen_global_step=0
            opt._durable_codebook_id=None
            model.train()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start=time.perf_counter()
            warmup=min(16,args.steps//4)
            report['measurement_warmup_updates']=warmup
            measured_start=None
            for tick,index in enumerate(indices):
                if tick==warmup:measured_start=time.perf_counter()
                t0=time.perf_counter()
                offset=int(index)*args.context
                ids=torch.tensor(data[offset:offset+args.context+1].astype(np.int64),device='cuda')[None]
                opt.zero_grad(set_to_none=True)
                with amp('cuda'):
                    _,loss=model(ids[:,:-1],ids[:,1:])
                    loss=loss+model.cfg.pool_aux*model.pool_aux()+model.pool_balance()
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not bool(torch.isfinite(loss)&torch.isfinite(norm)):raise AssertionError('nonfinite loss/gradient')
                if args.fingerprint and tick==0:
                    report['first_gradient_sha256']=fingerprint({n:p.grad for n,p in model.named_parameters()})
                torch.cuda.synchronize()
                opt_start=time.perf_counter()
                opt.step()
                torch.cuda.synchronize()
                row=dict(step=tick+1,objective=float(loss.detach()),gradient_norm=float(norm),
                         seconds=time.perf_counter()-t0,optimizer_seconds=time.perf_counter()-opt_start)
                report['trace'].append(row)
                if (tick+1)%16==0:print(json.dumps(row),flush=True)
            end=time.perf_counter()
            report.update(training_start=start,training_end=end,measurement_start=measured_start,
                          peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                          final_tune=evaluate(model,data,262144,args.context,16),
                          final_test=evaluate(model,data,524288,args.context,16))
            if args.fingerprint:
                report['final_model_sha256']=fingerprint(dict(model.state_dict()))
                report['final_optimizer_sha256']=fingerprint(opt.state_dict())
                report['final_host_sha256']=fingerprint(dict(pool.tiers.ram))
            if cache is not None:report['compute_cache']=dict(conversions=cache.conversions,hits=cache.hits)
            if pool.tiers.writebacks or pool.tiers.evictions or not pool.tiers.read_only:
                raise AssertionError('weights must remain cached and read-only')
            if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):raise AssertionError('nonfinite model')
            report.update(status='passed',final_slots=pool.slots,
                          deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
    except BaseException as exc:
        report.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('status','initial_test','final_test','peak_allocated_gib')}),flush=True)


if __name__=='__main__':main()
