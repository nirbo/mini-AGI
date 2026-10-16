"""Research-only, RAM-backed expert optimizer comparison; not a trainer backend.

Uses pinned Microsoft Dion algorithms with classic NS5 and the Muon+ author's
complete col_row optimizer. Never writes optimizer history
to the downloaded checkpoint. Non-expert parameters always use AdamW.
"""

from collections import defaultdict
import copy
from pathlib import Path
import subprocess
import sys
import weakref

import torch

from minagi.optimizers import NAMES, _move

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "dion": ("https://github.com/microsoft/dion.git", "7692479288f928a3d61d1cc58137b4dbe4f71000"),
    "MuonPlus": ("https://github.com/K1seki221/MuonPlus.git", "8a9ace123afedaab8ba75ea0b19315594ae1da7c"),
    "NorMuon": ("https://github.com/zichongli5/NorMuon.git", "c6989a8354730695d9f5a9faa6c55eeb24865209"),
}
VARIANTS = ("muon", "muon_plus", "normuon", "dion3_quarter", "dion3_half")


def source_path(name):
    path = ROOT / "runs" / "third_party" / name
    actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True).strip()
    if actual != SOURCES[name][1] or dirty:
        raise RuntimeError(f"{name}: expected clean pinned source {SOURCES[name][1]}, got {actual}")
    return path


def upstream():
    path = str(source_path("dion"))
    if path not in sys.path:
        sys.path.insert(0, path)
    import dion
    if Path(dion.__file__).resolve().parent.parent != Path(path):
        raise RuntimeError("unexpected Dion import location")
    return dion


@torch.compile(fullgraph=True, dynamic=False)
def classic_ns5(g, epsilon=1e-7):
    x = g.bfloat16()
    tall = g.size(-2) > g.size(-1)
    if tall:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + epsilon)
    for _ in range(5):
        a = x @ x.mT
        b = -4.7750 * a + 2.0315 * a @ a
        x = 3.4445 * x + b @ x
    return x.mT if tall else x


def upstream_plus():
    path = str(source_path('MuonPlus'))
    if path not in sys.path:
        sys.path.insert(0, path)
    from utils.optim import muon_plus
    if Path(muon_plus.__file__).resolve().parents[2] != Path(path):
        raise RuntimeError('unexpected MuonPlus import location')
    return muon_plus


def initialize_state(engine, p):
    if hasattr(engine, '_get_or_initialize_state'):
        return engine._get_or_initialize_state(p, engine._algo_name, engine.param_groups[0])
    engine.state[p].setdefault('use_muon', True)
    return engine.state[p]


def matrix_optimizer(variant, parameters, lr, weight_decay=.1):
    parameters = list(parameters)
    if variant == "muon_plus":
        return upstream_plus().MuonPlus([(f'expert{i}.weight', p) for i, p in enumerate(parameters)],
                                       lr=lr, weight_decay=weight_decay, norm_mode='col_row',
                                       polar_method='Keller', rms_scaling=True)
    dion = upstream()
    options = dict(lr=lr, weight_decay=weight_decay, mu=.95,
                   adjust_lr="rms_norm", epsilon=1e-7, use_polar_express=False,
                   use_triton=False, flatten=False, newton_schulz_func=classic_ns5)
    if variant == "muon":
        return dion.Muon(parameters, nesterov=True, **options)
    if variant == "normuon":
        return dion.NorMuon(parameters, nesterov=True, muon_beta2=.95, **options)
    if variant in ("dion3_quarter", "dion3_half"):
        return dion.Dion3(parameters, fraction=.25 if variant.endswith("quarter") else .5,
                          muon_beta2=.95, **options)
    raise ValueError(variant)


class ResearchPagedMatrix(torch.optim.Optimizer):
    """Read-only-checkpoint screen with lossless, UID-owned history in host RAM.

    This intentionally has no minagi checkpoint format or trainer integration.
    state_dict is a research snapshot of optimizer history only; weights and
    pager contents must be snapshotted separately. Topology changes are rejected.
    """

    def __init__(self, params, pool, variant, lr, weight_decay=.1, **kwargs):
        if not pool.tiers.read_only:
            raise ValueError("research adapter requires a read-only expert store")
        if any(getattr(pool, n).dtype != torch.float32 for n in NAMES):
            raise ValueError("FP32 masters required")
        self.variant = variant
        self._pool_ref = weakref.ref(pool)
        self._uids = tuple(pool.uid.tolist())
        self._adam = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, **kwargs)
        super().__init__(self._adam.param_groups, self._adam.defaults)
        self._views = [[torch.nn.Parameter(getattr(pool, n)[s].detach()) for n in NAMES]
                       for s in range(pool.resident)]
        self._matrix = matrix_optimizer(variant, [p for row in self._views for p in row], lr, weight_decay)
        self._owners = [None] * pool.resident
        self._history = {}
        self._counts = defaultdict(int)
        slabs = {id(getattr(pool, n)) for n in NAMES}
        if not slabs.issubset({id(p) for g in self.param_groups for p in g['params']}):
            raise ValueError("all expert slabs must be present")

    def export_expert_state(self, slot, uid):
        # Pool owns weights; this experiment keeps ALL optimizer history in RAM.
        # Do not put a new persistence format into production expert files.
        return {}

    def _park(self, slot):
        uid = self._owners[slot]
        if uid is not None:
            self._history[uid] = [_move(self._matrix.state[p], 'cpu') for p in self._views[slot]]

    def prepare_paged_step(self, pool):
        if pool is not self._pool_ref() or tuple(pool.uid.tolist()) != self._uids:
            raise RuntimeError("research screen does not support topology changes")
        incoming = [pool._f(e) if e >= 0 else None for e in pool.slots]
        # Park every outgoing owner before looking up any incoming owner:
        # a simultaneous A/B slot swap must not reset B before it is parked.
        for slot, uid in enumerate(incoming):
            if uid != self._owners[slot]:
                self._park(slot)
        for slot, position in enumerate(pool.slots):
            uid = pool._f(position) if position >= 0 else None
            for name, view in zip(NAMES, self._views[slot]):
                if view.data_ptr() != getattr(pool, name)[slot].data_ptr():
                    raise RuntimeError("expert storage changed")
            if uid == self._owners[slot]:
                continue
            saved = self._history.pop(uid, None)
            for j, p in enumerate(self._views[slot]):
                self._matrix.state[p] = _move(saved[j], p.device) if saved else {}
                initialize_state(self._matrix, p)
            self._owners[slot] = uid

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        pool = self._pool_ref()
        if pool._opt is not self:
            raise RuntimeError("attach optimizer before stepping")
        self.prepare_paged_step(pool)
        groups = {id(p): g for g in self.param_groups for p in g['params']}
        slab_groups = [groups[id(getattr(pool, n))] for n in NAMES]
        if len({(g['lr'], g['weight_decay']) for g in slab_groups}) != 1:
            raise ValueError("screen requires matching expert group options")
        self._matrix.param_groups[0].update(lr=slab_groups[0]['lr'], weight_decay=slab_groups[0]['weight_decay'])
        for s, row in enumerate(self._views):
            for n, p in zip(NAMES, row):
                slab = getattr(pool, n)
                p.grad = slab.grad[s] if self._owners[s] is not None and slab.grad is not None else None
        gradients = [(getattr(pool, n), getattr(pool, n).grad) for n in NAMES]
        try:
            self._matrix.step()
            for s, row in enumerate(self._views):
                for j, p in enumerate(row):
                    if p.grad is not None:
                        self._counts[(self._owners[s], NAMES[j])] += 1
            for p, _ in gradients:
                p.grad = None
            self._adam.param_groups, self._adam.state = self.param_groups, self.state
            self._adam.step()
        finally:
            for p, grad in gradients:
                p.grad = grad
            for row in self._views:
                for p in row:
                    p.grad = None
        return loss

    def state_dict(self):
        result = super().state_dict()
        history = _move(self._history, 'cpu')
        for s, uid in enumerate(self._owners):
            if uid is not None:
                history[uid] = [_move(self._matrix.state[p], 'cpu') for p in self._views[s]]
        result['research_matrix'] = dict(variant=self.variant, uids=self._uids, history=history,
                                         counts=dict(self._counts), groups=copy.deepcopy(self._matrix.state_dict()['param_groups']))
        return result

    def load_state_dict(self, saved):
        saved = copy.deepcopy(saved)
        extra = saved.pop('research_matrix')
        if extra['variant'] != self.variant or extra['uids'] != self._uids:
            raise ValueError("incompatible research optimizer snapshot")
        super().load_state_dict(saved)
        self._history, self._counts = _move(extra['history'], 'cpu'), defaultdict(int, extra['counts'])
        self._owners = [None] * len(self._owners)
        for current, previous in zip(self._matrix.param_groups, extra['groups']):
            current.update({k: v for k, v in previous.items() if k != 'params'})


def setup_sources():
    for name, (url, revision) in SOURCES.items():
        path = ROOT / 'runs' / 'third_party' / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(['git', 'clone', url, str(path)], check=True)
            subprocess.run(['git', '-C', str(path), 'checkout', '--detach', revision], check=True)
        source_path(name)


if __name__ == '__main__':
    setup_sources()
