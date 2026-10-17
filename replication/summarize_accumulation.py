"""Validate and summarize the matched accumulator experiment."""

import argparse
import json
from pathlib import Path
import statistics

from replication.run_accumulation_screen import BACKENDS
from replication.run_fp8_screen import validate


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path('runs/accum-screen'))
    ap.add_argument('--out',type=Path,default=Path('replication/results/accumulation.json'))
    args=ap.parse_args();root=args.root
    gate=json.loads((root/'repro-passed.json').read_text());assert gate['status']=='passed'
    repro=[]
    for backend in BACKENDS:
        pair=[]
        for folder in ('repro-a','repro-b'):
            r=json.loads((root/folder/f'{backend}-seed42-steps8.json').read_text());validate(r)
            assert len(r['trace'])==8
            pair.append(r);repro.append(r)
        for key in ('initial_model_sha256','first_gradient_sha256','final_model_sha256','final_optimizer_sha256',
                    'final_host_sha256','initial_tune','initial_test','final_tune','final_test','final_slots','training_window_indices'):
            assert pair[0][key]==pair[1][key],(backend,key)
        assert [(t['objective'],t['gradient_norm']) for t in pair[0]['trace']] == [(t['objective'],t['gradient_norm']) for t in pair[1]['trace']]
    fresh=json.loads((root/'repro-uncached/fp16-split4-seed42-steps8-uncached.json').read_text());validate(fresh)
    baseline=next(r for r in repro if r['configuration']['backend']=='fp16-split4')
    for key in ('first_gradient_sha256','final_model_sha256','final_optimizer_sha256','final_host_sha256','final_tune','final_test','final_slots'):
        assert fresh[key]==baseline[key],key
    runs=[]
    for seed in (42,43,44):
        paired=[]
        for backend in BACKENDS:
            r=json.loads((root/f'{backend}-seed{seed}-steps256.json').read_text());validate(r)
            cfg=r['configuration']
            assert (cfg['backend'],cfg['seed'],cfg['steps'],cfg['fingerprint'],cfg['no_cache'])==(backend,seed,256,False,False)
            assert len(r['trace'])==len(r['training_window_indices'])==256
            assert r['deterministic_algorithms']
            if seed==42:
                oracle=next(p for p in repro if p['configuration']['backend']==backend)
                assert [(t['objective'],t['gradient_norm']) for t in r['trace'][:8]] == [(t['objective'],t['gradient_norm']) for t in oracle['trace']]
            paired.append(r);runs.append(r)
        for key in ('initial_model_sha256','training_window_indices','manifest_sha256','text_sha256'):
            assert len({json.dumps(r[key],sort_keys=True) for r in paired})==1,key
    assert len({r['initial_model_sha256'] for r in runs+repro+[fresh]})==1
    learning={}
    for backend in BACKENDS:
        rows=[r for r in runs if r['configuration']['backend']==backend]
        periods={}
        for name,field in (('complete','external_training_power'),('after_16_updates','external_measured_training_power')):
            powers=[r[field] for r in rows]
            periods[name]=dict(mean_seconds=statistics.mean(p['duration_seconds'] for p in powers),
                               mean_new_tokens_per_second=statistics.mean(p['new_tokens_per_second'] for p in powers),
                               mean_watts=statistics.mean(p['average_board_power_w'] for p in powers),
                               peak_sampled_watts=max(p['peak_sampled_board_power_w'] for p in powers),
                               mean_joules_per_new_token=statistics.mean(p['estimated_board_joules_per_new_token'] for p in powers),
                               max_sample_gap_seconds=max(p['max_sample_gap_seconds'] for p in powers))
        learning[backend]=dict(initial_test_ce=rows[0]['initial_test']['mean_ce'],
                              seed_final_test_ce={str(r['configuration']['seed']):r['final_test']['mean_ce'] for r in rows},
                              mean_final_test_ce=statistics.mean(r['final_test']['mean_ce'] for r in rows),
                              max_allocated_gib=max(r['peak_allocated_gib'] for r in rows),periods=periods)
    probes=[]
    for repeat in (1,2):
        for phase in ('gemm','linear','experts'):
            for size in ('current','large'):
                r=json.loads((root/f'{phase}-{size}-repeat{repeat}.json').read_text());validate(r)
                expected=(12 if phase=='gemm' else 8) if size=='current' else 4
                if phase=='experts':expected=5
                assert len(r['cases'])==expected
                for c in r['cases']:
                    assert c['status']=='passed' and c['external_power']['sample_count_inside']>=20
                    probes.append(dict(phase=phase,size=size,repeat=repeat,**c))
    result=dict(status='passed',reproducibility=gate,learning=learning,learning_runs=runs,probes=probes,
                limits=['Three fixed-rate 256-update data-order seeds; not convergence or a hyperparameter search.',
                        'One undertrained checkpoint, depth 6, context 512, deterministic top-k routing.',
                        'Disjoint byte/marker-token ranges, not document-deduplicated; pretraining overlap unknown.',
                        'Evaluation uses each backend arithmetic; no common-arithmetic learned-weight comparison.',
                        'FP16 operands, internal MMA accumulation and outer FP32 gradient storage are distinct.',
                        'Research backends share padded dispatch; Sonic is fused/sparse.',
                        'Cached synthetic weights stay unchanged; model runs include actual updates/paging.',
                        'NVML sensor averaging and desktop activity affect board energy; CPU energy excluded.'])
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    for name,row in learning.items():print(name,json.dumps(row),flush=True)


if __name__=='__main__':main()
