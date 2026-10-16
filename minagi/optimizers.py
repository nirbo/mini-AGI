"""Optional Gefen experts + AdamW auxiliary parameters, with UID-owned state.

Only resident matrices live in Gefen. Its shared codebook is learned once;
each matrix's counter advances on resident updates, including zero gradients,
and pauses while evicted. Missing slab gradients and empty slots are skipped.
The FP32 slab remains the autograd parameter; independent 2D views share its
storage only for optimizer updates. No expert axis is flattened into a matrix.
"""

from collections import defaultdict
import copy
import hashlib
import importlib.metadata
import math
import weakref

import torch

from . import optim_codec

EXPERT_PREFIX = "__gefen_"
KIND = "gefen-experts-adamw-v1"
NAMES = ("w1", "w3", "w2")


def _move(tree, device):
    if torch.is_tensor(tree):
        return tree.detach().to(device=device).clone()
    if isinstance(tree, dict):
        return {k: _move(v, device) for k, v in tree.items()}
    if isinstance(tree, (tuple, list)):
        return type(tree)(_move(v, device) for v in tree)
    return copy.deepcopy(tree)


class PagedGefen(torch.optim.Optimizer):
    """Gefen 1.2.3 expert state adapter; AdamW stays the auxiliary optimizer."""

    def __init__(self, params, pool, lr=1e-3, betas=(.9, .95), weight_decay=0.,
                 eps=1e-8, fused=None):
        try:
            version = importlib.metadata.version("gefen")
        except importlib.metadata.PackageNotFoundError as exc:
            raise ImportError("optional Gefen 1.2.3 must be installed in the repository .venv") from exc
        if version != "1.2.3":
            raise RuntimeError(f"Gefen state adapter requires qualified version 1.2.3, got {version}")
        if not hasattr(pool, "tiers"):
            raise ValueError("Gefen expert paging requires PagedPool")
        if any(getattr(pool, name).dtype != torch.float32 for name in NAMES):
            raise ValueError("Gefen expert paging requires FP32 masters")
        from gefen import Gefen

        self._pool_ref = weakref.ref(pool)
        fused = pool.w1.device.type == "cuda" if fused is None else fused
        adam = torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay,
                                 eps=eps, fused=fused)
        super().__init__(adam.param_groups, adam.defaults)
        self._adam = adam
        self._views = [[torch.nn.Parameter(getattr(pool, name)[s].detach()) for name in NAMES]
                       for s in range(pool.resident)]
        self._gefen = Gefen([p for row in self._views for p in row], lr=lr, betas=betas,
                            weight_decay=weight_decay, eps=eps, fused=fused)
        self._sync_adam()
        if not all(any(p is getattr(pool, name) for g in self.param_groups for p in g["params"])
                   for name in NAMES):
            raise ValueError("optimizer must contain all three expert slabs")

    def _sync_adam(self):
        self._adam.param_groups = self.param_groups
        self._adam.state = self.state

    def uses_gefen_parameter(self, parameter):
        pool = self._pool_ref()
        return pool is not None and any(parameter is getattr(pool, name) for name in NAMES)

    def _global(self):
        return {"codebook": _move(self._gefen._gefen_codebook, "cpu"),
                "step": self._gefen._gefen_global_step}

    def _codebook_id(self):
        codebook = self._gefen._gefen_codebook
        if codebook is None:
            return None
        cached = getattr(self, "_codebook_cache", None)
        if cached is None or cached[0] is not codebook or cached[1] != codebook._version:
            self._codebook_cache = (codebook, codebook._version,
                                   hashlib.sha256(codebook.detach().cpu().numpy().tobytes()).hexdigest())
        return self._codebook_cache[2]

    def _validate_expert_state(self, state, shape):
        if not state:
            return
        count = math.prod(shape)
        period, step = state.get("automatic_period"), state.get("step")
        if type(period) is not int or period <= 0 or count % period or type(step) is not int or step < 1:
            raise ValueError("invalid Gefen partition or counter")
        partition_shape = (count // period, period)
        if state.get("m_codebook_shape") != partition_shape:
            raise ValueError("invalid Gefen index shape metadata")
        magnitude_rows = count // period
        if period == 1:
            divisor = state.get("period_one_divisor")
            if type(divisor) is not int or divisor <= 0 or count % divisor:
                raise ValueError("invalid Gefen magnitude partition")
            magnitude_rows = count // divisor
        expected = {"m_codebook": (torch.uint8, partition_shape),
                    "m_magnitude": (torch.float32, (magnitude_rows, 1)),
                    "vmean": (torch.float32, (count // period, 1))}
        for key, (dtype, tensor_shape) in expected.items():
            tensor = state.get(key)
            if not torch.is_tensor(tensor) or tensor.dtype != dtype or tuple(tensor.shape) != tensor_shape:
                raise ValueError(f"invalid Gefen tensor {key}")
            if tensor.is_floating_point() and (not bool(torch.isfinite(tensor).all()) or bool((tensor < 0).any())):
                raise ValueError(f"invalid Gefen values in {key}")

    def _restore_global(self, value):
        device = self._pool_ref().w1.device
        codebook = value["codebook"]
        if codebook is not None and (codebook.dtype != torch.float32 or codebook.shape != (256,)
                                     or not bool(torch.isfinite(codebook).all())
                                     or not bool((codebook[1:] >= codebook[:-1]).all())):
            raise ValueError("invalid Gefen codebook")
        if type(value["step"]) is not int or value["step"] < 0:
            raise ValueError("invalid Gefen global step")
        self._gefen._gefen_codebook = _move(codebook, device)
        self._gefen._gefen_global_step = value["step"]

    def export_expert_state(self, slot, uid):
        states = {name: _move(self._gefen.state.get(p, {}), "cpu")
                  for name, p in zip(NAMES, self._views[slot])}
        payload = {"kind": KIND, "uid": uid, "codebook_id": self._codebook_id(),
                   "shapes": [tuple(p.shape) for p in self._views[slot]], "states": states}
        return {k: torch.from_numpy(v) for k, v in optim_codec.encode(payload, EXPERT_PREFIX).items()}

    def prepare_paged_step(self, pool):
        if self._pool_ref() is not pool:
            raise ValueError("Gefen optimizer attached to a different pool")
        for s, e in enumerate(pool.slots):
            for name, p in zip(NAMES, self._views[s]):
                slab = getattr(pool, name)
                if slab.dtype != torch.float32 or p.data_ptr() != slab[s].data_ptr():
                    raise RuntimeError("expert storage changed; rebuild Gefen after moving the model")
            if pool._own[s]:
                continue
            states = {}
            if e >= 0:
                uid = pool._f(e)
                entry = pool.tiers.fetch(uid, count=False)
                if EXPERT_PREFIX + "json" in entry:
                    payload = optim_codec.decode(entry, EXPERT_PREFIX)
                    if (payload["kind"] != KIND or payload["uid"] != uid or
                            payload["shapes"] != [tuple(p.shape) for p in self._views[s]]):
                        raise ValueError("Gefen expert state identity/geometry mismatch")
                    states = payload["states"]
                    if set(states) != set(NAMES):
                        raise ValueError("invalid Gefen expert projection membership")
                    if any(st.get("step", 0) for st in states.values()) and payload["codebook_id"] != self._codebook_id():
                        raise ValueError("Gefen expert history requires its matching optimizer codebook; restore optim.npz first")
                    for name, p in zip(NAMES, self._views[s]):
                        self._validate_expert_state(states[name], tuple(p.shape))
            for name, p in zip(NAMES, self._views[s]):
                self._gefen.state[p] = _move(states.get(name, {}), p.device)
            pool._own[s] = e >= 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        pool = self._pool_ref()
        if pool._opt is not self:
            raise RuntimeError("attach Gefen to its pool before stepping")
        # A closure can run a forward and change admission. Restore afterwards.
        self.prepare_paged_step(pool)
        groups = {id(p): g for g in self.param_groups for p in g["params"]}
        for name in NAMES:
            group = groups[id(getattr(pool, name))]
            if any(group.get(key, False) for key in ("amsgrad", "maximize", "differentiable", "capturable")):
                raise ValueError("Gefen expert groups do not support AMSGrad, maximize, differentiable or capturable")
            grad = getattr(pool, name).grad
            if grad is not None and grad.is_sparse:
                raise ValueError("Gefen expert gradients must be dense")
        active = False
        for s, row in enumerate(self._views):
            for name, view in zip(NAMES, row):
                slab = getattr(pool, name)
                group = groups[id(slab)]
                engine_group = self._gefen.param_groups[s * 3 + NAMES.index(name)]
                engine_group.update(lr=group["lr"], weight_decay=group["weight_decay"],
                                    beta1=group["betas"][0], beta2=group["betas"][1], eps=group["eps"])
                view.grad = slab.grad[s] if pool.slots[s] >= 0 and slab.grad is not None else None
                active |= view.grad is not None
        try:
            if active:
                self._gefen.step()
            gradients = [(getattr(pool, name), getattr(pool, name).grad) for name in NAMES]
            for slab, _ in gradients:
                slab.grad = None
            try:
                self._sync_adam()
                self._adam.step()
            finally:
                for slab, grad in gradients:
                    slab.grad = grad
        finally:
            for row in self._views:
                for view in row:
                    view.grad = None
        return loss

    def state_dict(self):
        result = super().state_dict()
        pool = self._pool_ref()
        result["paged_gefen"] = {"kind": KIND, "global": self._global(),
                                  "experts": {pool._f(e): self.export_expert_state(s, pool._f(e))
                                              if pool._own[s] else {
                                                  k: v.clone() for k, v in pool.tiers.fetch(pool._f(e), count=False).items()
                                                  if k.startswith(EXPERT_PREFIX)}
                                              for s, e in enumerate(pool.slots) if e >= 0}}
        return result

    def load_state_dict(self, state_dict):
        saved = dict(state_dict)
        extra = saved.pop("paged_gefen")
        if extra["kind"] != KIND:
            raise ValueError("unsupported Gefen optimizer format")
        super().load_state_dict(saved)
        self._restore_global(extra["global"])
        pool = self._pool_ref()
        for uid, arrays in extra["experts"].items():
            entry = dict(pool.tiers.fetch(uid, count=False))
            entry = {k: v for k, v in entry.items() if not k.startswith(EXPERT_PREFIX)}
            entry.update(_move(arrays, "cpu"))
            pool.tiers.put(uid, entry)
        pool._own = [False] * pool.resident
        self._sync_adam()

    def checkpoint_arrays(self, model):
        names = {id(p): name for name, p in model.named_parameters()}
        groups = [{"params": [names[id(p)] for p in group["params"]],
                   "options": {k: v for k, v in group.items() if k != "params"}}
                  for group in self.param_groups]
        states = {names[id(p)]: _move(state, "cpu") for p, state in self.state.items()}
        return optim_codec.encode({"kind": KIND, "global": self._global(),
                                   "groups": groups, "states": states})

    def restore_checkpoint(self, model, arrays):
        payload = optim_codec.decode(arrays)
        if payload["kind"] != KIND:
            raise ValueError("unsupported Gefen optimizer checkpoint")
        parameters = dict(model.named_parameters())
        saved_names = [name for group in payload["groups"] for name in group["params"]]
        if len(saved_names) != len(set(saved_names)) or set(saved_names) != set(parameters):
            raise ValueError("Gefen checkpoint parameter membership differs from model")
        groups = [{**group["options"], "params": [parameters[n] for n in group["params"]]}
                  for group in payload["groups"]]
        states = defaultdict(dict)
        for name, state in payload["states"].items():
            p = parameters[name]
            if self.uses_gefen_parameter(p):
                raise ValueError("Gefen expert state must be stored by UID, not slab")
            for key in ("exp_avg", "exp_avg_sq"):
                if key in state and state[key].shape != p.shape:
                    raise ValueError(f"optimizer checkpoint shape mismatch for {name}")
            states[p] = _move(state, p.device)
        self._restore_global(payload["global"])
        self.param_groups, self.state = groups, states
        self._pool_ref()._own = [False] * self._pool_ref().resident
        self._sync_adam()


def make_optimizer(params, *, pool=None, optimizer="adamw", **kwargs):
    if optimizer == "adamw":
        return torch.optim.AdamW(params, **kwargs)
    if optimizer == "gefen":
        return PagedGefen(params, pool, **kwargs)
    raise ValueError(f"unknown optimizer: {optimizer!r}")
