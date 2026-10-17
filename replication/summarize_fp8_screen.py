"""Validate FP8/Gefen experiment provenance and summarize learning/timings."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics

from replication.run_fp8_screen import validate


def summarize(root):
    gate=json.loads((root/'repro-passed.json').read_text())
    if gate.get('status')!='passed':raise AssertionError('replay gate failed')
    repro=[]
    for folder,backends in (('repro-a',('sonic','fp8')),('repro-b',('sonic','fp8')),
                            ('repro-uncached',('fp8',))):
        for backend in backends:
            suffix='-uncached' if folder=='repro-uncached' else ''
            r=json.loads((root/folder/f'{backend}-seed42-steps8{suffix}.json').read_text())
            validate(r);repro.append(r)
    if len({r['initial_model_sha256'] for r in repro})!=1:raise AssertionError('unequal replay initial weights')
    for backend in ('sonic','fp8'):
        trials=[r for r in repro if r['configuration']['backend']==backend]
        for key in ('first_gradient_sha256','final_model_sha256','final_optimizer_sha256',
                    'final_host_sha256','initial_test','initial_tune','final_test','final_tune','final_slots'):
            if any(r[key]!=trials[0][key] for r in trials):raise AssertionError(f'replay mismatch: {backend} {key}')
    learning=[]
    for seed in (42,43,44):
        paired=[]
        for backend in ('sonic','fp8'):
            r=json.loads((root/f'{backend}-seed{seed}-steps256.json').read_text())
            validate(r);cfg=r['configuration']
            if (cfg['backend'],cfg['seed'],cfg['steps'],cfg['fingerprint'],cfg['no_cache'])!=(backend,seed,256,False,False):
                raise AssertionError('unexpected learning configuration')
            if len(r['trace'])!=256 or len(r['training_window_indices'])!=256:raise AssertionError('incomplete trial')
            if not r['deterministic_algorithms']:raise AssertionError('uncontrolled trial')
            if seed==42:
                oracle=next(p for p in repro if p['configuration']['backend']==backend)
                for a,b in zip(r['trace'][:8],oracle['trace']):
                    if any(a[k]!=b[k] for k in ('objective','gradient_norm')):raise AssertionError('changed replay prefix')
            paired.append(r);learning.append(r)
        for key in ('initial_model_sha256','training_window_indices','manifest_sha256','text_sha256'):
            if paired[0][key]!=paired[1][key]:raise AssertionError(f'unmatched pair: {key}')
    if len({r['initial_model_sha256'] for r in learning})!=1:raise AssertionError('seed changes initial model')
    summary={}
    for backend in ('sonic','fp8'):
        rows=[r for r in learning if r['configuration']['backend']==backend]
        periods={}
        for name,field in (('complete','external_training_power'),('after_16_updates','external_measured_training_power')):
            powers=[r[field] for r in rows]
            periods[name]=dict(mean_seconds=statistics.mean(p['duration_seconds'] for p in powers),
                               mean_new_tokens_per_second=statistics.mean(p['new_tokens_per_second'] for p in powers),
                               mean_watts=statistics.mean(p['average_board_power_w'] for p in powers),
                               peak_sampled_watts=max(p['peak_sampled_board_power_w'] for p in powers),
                               mean_joules_per_new_token=statistics.mean(p['estimated_board_joules_per_new_token'] for p in powers),
                               max_sample_gap_seconds=max(p['max_sample_gap_seconds'] for p in powers))
        summary[backend]=dict(initial_test_ce=rows[0]['initial_test']['mean_ce'],
                             seed_final_test_ce={str(r['configuration']['seed']):r['final_test']['mean_ce'] for r in rows},
                             mean_final_test_ce=statistics.mean(r['final_test']['mean_ce'] for r in rows),
                             max_allocated_gib=max(r['peak_allocated_gib'] for r in rows),periods=periods)
    bf16=summary['sonic']['seed_final_test_ce'];fp8=summary['fp8']['seed_final_test_ce']
    paired_deltas={seed:fp8[seed]-value for seed,value in bf16.items()}
    probes=[]
    for repeat in (1,2):
        for phase in ('linear','experts'):
            for size in ('current','large'):
                r=json.loads((root/f'{phase}-{size}-repeat{repeat}.json').read_text());validate(r)
                expected=24 if phase=='linear' and size=='current' else 6
                if len(r['cases'])!=expected:raise AssertionError('incomplete timing probe')
                for case in r['cases']:
                    if case['status']!='passed' or 'external_power' not in case:raise AssertionError('unqualified timing case')
                    probes.append(dict(phase=phase,size=size,repeat=repeat,**case))
    return dict(status='passed',reproducibility=gate,learning_trial_count=6,
                learning=summary,paired_test_ce_fp8_minus_bf16=paired_deltas,
                probes=probes,learning_runs=learning,
                limits=['Three fixed-rate 256-update BF16/FP8 pairs on one existing checkpoint, not convergence proof.',
                        'Disjoint token ranges in TinyStories validation text; pretraining overlap unknown.',
                        'Depth 6 and context 512; default deterministic top-k routing.',
                        'FP8 prototype is padded and has separate FP32 output rescaling; Sonic BF16 is fused/sparse.',
                        'Cached synthetic probes reuse unchanged weights; complete model runs include actual updates and paging.',
                        'Synthetic timing energy is per input row/token; only model runs measure new language-model tokens.',
                        'Deterministic execution controls; power includes desktop activity and sensor averaging.'])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path('runs/fp8-screen'))
    ap.add_argument('--out',type=Path,default=Path('replication/results/fp8_gefen.json'))
    args=ap.parse_args();r=summarize(args.root)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(r,indent=2)+'\n')
    for name,row in r['learning'].items():print(name,json.dumps({k:v for k,v in row.items()}),flush=True)
    print('Paired test CE deltas (FP8 - BF16):',r['paired_test_ce_fp8_minus_bf16'])


if __name__=='__main__':main()
