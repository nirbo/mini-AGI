"""Locate backward/norm variability using repeated backward on one fixed graph."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import types

import numpy as np
import torch

from minagi.ingest import as_stream
from minagi.pool import PooledMLP
from minagi.precision import amp,set_compute_dtype
from minagi.recur import load_any
from replication.compare_gefen_training import evaluate,weights_only


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--backend',choices=('sonic','reference'),default='sonic')
    ap.add_argument('--deterministic',action='store_true')
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    # These are process-local execution controls, not system/GPU changes.
    if args.deterministic:
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.cuda.set_per_process_memory_fraction(.75)
    torch.backends.cuda.matmul.allow_tf32=False
    set_compute_dtype('bf16')
    report={'backend':args.backend,'deterministic':args.deterministic}
    try:
        model,_=load_any('/mnt/fast/models/mini-AGI/undertrained/weights',torch.device('cuda'),read_only=True)
        model.cfg.max_steps=model.cfg.min_steps=model.cfg.bptt_window=6
        model.cfg.train_steps_mean=0
        for site in model.modules():
            if isinstance(site,PooledMLP):
                site.expert_backend=args.backend
                site.grad_checkpoint=True
        pool=model.pool
        pool.tiers.ram_capacity=pool.n_experts()
        pool.tiers._from_disk=types.MethodType(weights_only,pool.tiers)
        data=as_stream('/mnt/fast/models/mini-AGI/evaluation-data/TinyStoriesV2-GPT4-valid.txt')
        evaluate(model,data,262144,512,16)
        evaluate(model,data,524288,512,16)
        index=int(np.random.default_rng(42).permutation(511)[0])
        ids=torch.tensor(data[index*512:index*512+513].astype(np.int64),device='cuda')[None]
        model.train()
        with amp('cuda'):
            _,loss=model(ids[:,:-1],ids[:,1:])
            loss=loss+model.cfg.pool_aux*model.pool_aux()+model.pool_balance()
        report['objective']=float(loss.detach())
        first=None
        report['backwards']=[]
        from torch.nn.utils.clip_grad import _get_total_norm
        for repeat in range(3):
            model.zero_grad(set_to_none=True)
            loss.backward(retain_graph=True)
            grads=[p.grad for p in model.parameters() if p.grad is not None]
            row={'repeat':repeat,
                 'foreach_norms':[float(_get_total_norm(grads,foreach=True)) for _ in range(4)],
                 'individual_norms':[float(_get_total_norm(grads,foreach=False)) for _ in range(4)]}
            current={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.grad is not None}
            row['gradient_sha256']={n:hashlib.sha256(g.numpy().tobytes()).hexdigest() for n,g in current.items()}
            if first is None:
                first=current
            else:
                row['gradient_differences']={n:{'max_abs':float((g-first[n]).abs().max()),
                                               'relative_l2':float((g-first[n]).norm()/first[n].norm().clamp_min(1e-30))}
                                             for n,g in current.items() if not torch.equal(g,first[n])}
            report['backwards'].append(row)
        report['repeatable_within_graph']=all(b['gradient_sha256']==report['backwards'][0]['gradient_sha256']
                                              for b in report['backwards'][1:])
        if args.deterministic and not report['repeatable_within_graph']:
            raise AssertionError('deterministic repeated backward changed gradients')
        report['status']='passed'
    except BaseException as exc:
        report.update(status='failed',error=repr(exc))
        raise
    finally:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('status','backend','deterministic','objective','repeatable_within_graph')}),flush=True)


if __name__=='__main__':
    main()
