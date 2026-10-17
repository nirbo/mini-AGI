"""Validate the matched experiment and emit compact, auditable result tables."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics

from replication.muon_candidates import VARIANTS, SOURCES
from replication.run_muon_screen import validate_learning_protocol


def summarize(root):
    variants = ('adamw', 'gefen', *VARIANTS)
    reports = [json.loads(p.read_text()) for p in sorted(root.glob('*-seed*-steps*.json'))
               if re.fullmatch(r'.+-lr.+-seed\d+-steps\d+\.json',p.name)]
    if any(r['status'] != 'passed' for r in reports):
        raise AssertionError('failed learning trial')
    if not reports:
        raise AssertionError('no trials')
    for r in reports:
        validate_learning_protocol(r,False)
        if any(len(r[key])!=r['configuration']['steps'] for key in ('trace','training_window_indices')):
            raise AssertionError('incomplete learning run')
        if any(t['routing_seed'] is not None for t in r['trace']):
            raise AssertionError('this screen requires the default deterministic top-k routing')
        for source, digest in r['sources'].items():
            if hashlib.sha256(Path(source).read_bytes()).hexdigest() != digest:
                raise AssertionError(f'source changed since run: {source}')
    initial_tune = reports[0]['initial_tune']
    initial_test = reports[0]['initial_test']
    for r in reports:
        if r['initial_tune'] != initial_tune or r['initial_test'] != initial_test:
            raise AssertionError('unequal initial held-out results')
    grouped = {}
    for r in reports:
        cfg = r['configuration']
        key = (cfg['steps'], cfg['seed'])
        indices = grouped.setdefault(key, r['training_window_indices'])
        if indices != r['training_window_indices']:
            raise AssertionError('unmatched data order')
        if r['manifest_sha256'] != reports[0]['manifest_sha256'] or r['text_sha256'] != reports[0]['text_sha256']:
            raise AssertionError('different checkpoint or data')
    prefixes = []
    short_runs = {(r['configuration']['optimizer'],r['configuration']['lr']):r for r in reports
                  if r['configuration']['seed']==42 and r['configuration']['steps']==64}
    for r in reports:
        cfg = r['configuration']
        if cfg['seed']!=42 or cfg['steps']!=256:
            continue
        earlier=short_runs.get((cfg['optimizer'],cfg['lr']))
        if earlier is None:
            continue
        if r['training_window_indices'][:64] != earlier['training_window_indices']:
            raise AssertionError('long/short data prefixes differ')
        prefixes.append(dict(optimizer=cfg['optimizer'],lr=cfg['lr'],
                             max_objective_abs_diff=max(abs(a['objective']-b['objective']) for a,b in zip(r['trace'][:64],earlier['trace'])),
                             max_gradient_norm_abs_diff=max(abs(a['gradient_norm_before_clip']-b['gradient_norm_before_clip']) for a,b in zip(r['trace'][:64],earlier['trace']))))
    long_selection = (root/'selected-rates-long.json').exists()
    horizon = 256 if long_selection else 64
    selected = json.loads((root/('selected-rates-long.json' if long_selection else 'selected-rates.json')).read_text())
    table, grid, short_grid = {}, {}, {}
    for name in variants:
        short = [r for r in reports if r['configuration']['optimizer'] == name
                 and r['configuration']['steps'] == 64 and r['configuration']['seed'] == 42]
        if horizon==64 and len(short) < 6:
            raise AssertionError(f'incomplete rate grid: {name}')
        short_grid[name] = sorted([dict(lr=r['configuration']['lr'], tune_ce=r['final_tune']['mean_ce'],
                                  test_ce=r['final_test']['mean_ce']) for r in short], key=lambda r:r['lr'])
        selection_runs = [r for r in reports if r['configuration']['optimizer']==name
                          and r['configuration']['steps']==horizon and r['configuration']['seed']==42]
        grid[name] = sorted([dict(lr=r['configuration']['lr'], tune_ce=r['final_tune']['mean_ce'],
                                 test_ce=r['final_test']['mean_ce']) for r in selection_runs],key=lambda r:r['lr'])
        winner = min(grid[name], key=lambda r:r['tune_ce'])
        if selected[name] != winner['lr']:
            raise AssertionError('rate was not selected on tuning loss')
        if winner['lr'] != 0 and winner['lr'] in (grid[name][0]['lr'], grid[name][-1]['lr']):
            raise AssertionError(f'unbracketed winning rate: {name}')
        long = [r for r in reports if r['configuration']['optimizer'] == name
                and r['configuration']['steps'] == 256 and r['configuration']['seed'] in (43,44)
                and r['configuration']['lr']==selected[name]]
        if {r['configuration']['seed'] for r in long} != {43,44} or len(long) != 2:
            raise AssertionError(f'incomplete confirmation: {name}')
        if any(r['configuration']['lr'] != selected[name] for r in long):
            raise AssertionError('confirmation rate changed')
        power = [r['external_training_power'] for r in long]
        table[name] = dict(lr=selected[name],
                          experts_frozen=selected[name]==0,
                          mean_test_ce=statistics.mean(r['final_test']['mean_ce'] for r in long),
                          mean_tune_ce=statistics.mean(r['final_tune']['mean_ce'] for r in long),
                          seed_test_ce={r['configuration']['seed']:r['final_test']['mean_ce'] for r in long},
                          seed_tune_ce={r['configuration']['seed']:r['final_tune']['mean_ce'] for r in long},
                          mean_training_seconds=statistics.mean(p['duration_seconds'] for p in power),
                          mean_tokens_per_second=statistics.mean(p['new_tokens_per_second'] for p in power),
                          mean_board_watts=statistics.mean(p['average_board_power_w'] for p in power),
                          max_sampled_board_watts=max(p['peak_sampled_board_power_w'] for p in power),
                          mean_board_joules_per_token=statistics.mean(p['estimated_board_joules_per_new_token'] for p in power),
                          max_allocated_gib=max(r['peak_allocated_gib'] for r in long),
                          max_external_sample_gap_seconds=max(p['max_sample_gap_seconds'] for p in power),
                          mean_last16_optimizer_ms=statistics.mean(t['optimizer_seconds']*1000 for r in long for t in r['trace'][-16:]),
                          mean_last16_full_step_ms=statistics.mean(t['seconds']*1000 for r in long for t in r['trace'][-16:]))
    latency = {name:json.loads((root/f'latency-{name}.json').read_text()) for name in variants}
    if any(not r['finite'] for r in latency.values()):
        raise AssertionError('non-finite optimizer benchmark')
    large = {name:json.loads((root/f'latency-large-{name}.json').read_text()) for name in variants}
    for r in (*latency.values(),*large.values()):
        if r.get('status')=='out_of_memory':
            continue
        if not r['finite'] or not r['deterministic_algorithms'] or r['sources']!={k:list(v) for k,v in SOURCES.items()}:
            raise AssertionError('unqualified optimizer-only benchmark')
    frozen = [r for r in reports if r['configuration']['optimizer']=='adamw' and r['configuration']['lr']==0.
              and r['configuration']['steps']==256]
    if {r['configuration']['seed'] for r in frozen}!={42,43,44} or len(frozen)!=3:
        raise AssertionError('incomplete matched auxiliary-only controls')
    control={r['configuration']['seed']:r['final_test']['mean_ce'] for r in frozen}
    for name,row in table.items():
        row['seed_test_ce_minus_adamw']={seed:value-table['adamw']['seed_test_ce'][seed]
                                         for seed,value in row['seed_test_ce'].items()}
        row['seed_test_ce_minus_auxiliary_only']={seed:value-control[seed]
                                                 for seed,value in row['seed_test_ce'].items()}
    return dict(sources=SOURCES, learning_trial_count=len(reports),
                manifest_sha256=reports[0]['manifest_sha256'], text_sha256=reports[0]['text_sha256'],
                initial_tune=initial_tune, initial_test=initial_test, grid=grid, short_grid=short_grid,
                selection_horizon_steps=horizon, confirmation=table,
                long_short_prefix_checks=prefixes,
                reproducibility=json.loads(Path('runs/muon-deterministic-repro-fixed/repro-passed.json').read_text()),
                isolated_latency=latency, isolated_latency_large=large,
                auxiliary_only_control=frozen,
                measured_source_sha256=reports[0]['sources'],
                limits=['Short BF16 adaptation, not original-depth pretraining or NVFP4 qualification.',
                        'Two data-order seeds, default deterministic top-k routing, and disjoint token ranges in one validation text; original pretraining overlap unknown.',
                        'Available implementations differ in batching and shape scaling; timings are not algorithm-only comparisons.',
                        'Learning-dependent routing changes the work; optimizer-only measurements isolate available implementations.',
                        'Board energy includes desktop activity and sampling limitations.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('runs/muon-screen'))
    parser.add_argument('--out', type=Path, default=Path('replication/results/muon_variants.json'))
    args=parser.parse_args()
    report=summarize(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print('optimizer | LR | test CE | training s | avg W | peak W | J/token | peak GiB')
    for name,r in report['confirmation'].items():
        print(f"{name} | {r['lr']:g} | {r['mean_test_ce']:.5f} | {r['mean_training_seconds']:.2f} | "
              f"{r['mean_board_watts']:.1f} | {r['max_sampled_board_watts']:.1f} | "
              f"{r['mean_board_joules_per_token']:.5f} | {r['max_allocated_gib']:.3f}")


if __name__ == '__main__':
    main()
