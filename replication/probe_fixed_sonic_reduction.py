"""Qualify the experiment's fixed Sonic reduction against the PyTorch oracle."""

import json

from replication.compare_muon_deterministic import pin_sonic_reduction
import torch

from minagi.expert_compute import compute_experts
from minagi.precision import amp,set_compute_dtype


def main():
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_num_threads(4)
    torch.manual_seed(42)
    set_compute_dtype('bf16')
    config=pin_sonic_reduction()
    x=torch.randn(17,512,device='cuda',requires_grad=True)
    weights=tuple((torch.randn(*s,device='cuda')/s[-1]**.5).requires_grad_()
                  for s in ((32,64,512),(32,64,512),(32,512,64)))
    pairs=[(t,e) for t in range(17) if t not in (3,5) for e in (t,t+1)]
    tokens,experts=torch.tensor(pairs,device='cuda').T
    scores=(torch.randn(len(pairs),device='cuda')*.2)
    scores[0]=0
    scores.requires_grad_()
    upstream=torch.randn_like(x)
    results=[]
    for backend,recompute in (('reference',False),('sonic',False),('sonic',True)):
        with amp('cuda'):
            y=compute_experts(x,[weights],experts,tokens,scores,backend=backend,recompute=recompute)
        grads=torch.autograd.grad(y,[x,*weights,scores],upstream)
        assert all(bool(torch.isfinite(g).all()) for g in grads)
        assert int(y[[3,5]].count_nonzero())==0
        assert abs(float(grads[-1][0]))>0
        for g in grads[1:4]: assert int(g[18:].count_nonzero())==0
        results.append((y,grads))
    errors=[]
    for y,grads in results[1:]:
        actual=[y,*grads]
        expected=[results[0][0],*results[0][1]]
        relative=[float((a.detach()-b.detach()).float().norm()/b.detach().float().norm().clamp_min(1e-30)) for a,b in zip(actual,expected)]
        assert max(relative)<.025,relative
        errors.append(relative)
    torch.testing.assert_close(results[1][0],results[2][0],rtol=0,atol=0)
    for a,b in zip(results[1][1],results[2][1]): torch.testing.assert_close(a,b,rtol=0,atol=0)
    print(json.dumps({'status':'passed','fixed_config':config,'relative_errors_output_x_w1_w3_w2_scores':errors,
                      'recompute_bit_identical':True,'missing_tokens_unused_experts_zero_score_derivative':True}),flush=True)


if __name__=='__main__':
    main()
