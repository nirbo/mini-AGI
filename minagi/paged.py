"""
A pool larger than the card it runs on.

The claim this exists to make true: disk holds every expert, RAM caches the
ones recently wanted, and VRAM holds only the ones being worked with now. Three
tiers, and only the last is scarce.

CAUSAL WORKING SETS

The first recurrent router asks for experts in token order. Each sequence adds
new requests until its resident budget is full, then keeps that set until the
context resets. A position may use only experts introduced by its own prefix;
experts loaded for later positions are masked out, at every recurrent depth.

Cached chunks continue the same working set. Multiple sequences have independent
budgets (resident // batch_size), so one sample cannot take another's slots.
No chosen expert is evicted inside a text, preserving weights through backward
and optimizer updates. This changes the previous whole-chunk voting rule:
training and evaluation no longer receive future information through admission.

At selection temperature zero requests and routes are top-k. At a positive
temperature Gumbel draws are keyed by sequence, position, site and depth, so
changing chunk size does not change a token's random draw. The balance loss
still encourages use of the full pool across training texts.

WHAT MAKES PAGING HARD is not the weights. It is the optimiser. Adam keeps two
moments per parameter, and those belong to the EXPERT, not to the slot of VRAM
it happens to occupy. Stepping an expert with the moments its slot held for the
expert before it would hand it a stranger's momentum - the model would keep
training, and every paged expert would be steered by someone else's history.
So every expert on the card is stepped with its own moments, and the optimiser
is told about the slots rather than about the experts.

Only a step reads them, so they travel only to a step: an expert comes to the
card with its weights alone, and just before each optimiser step
(_own_moments) every expert on the card that is not already holding its own
moments gets them from its entry in RAM or on disk. A reply loads experts at
almost every character and steps none of them, so writing moves no moments at
all; parking an expert that was stepped carries its moments back with its
weights.
"""

import hashlib
import itertools
import math
import os
import struct
import weakref
import zipfile
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from numpy.lib import format as npy

from .precision import is_moment, pack_bf16, unpack_bf16

_WEIGHTS = ("w1", "w3", "w2")
_MOMENTS = ("w1_m", "w3_m", "w2_m", "w1_v", "w3_v", "w2_v")
_ZIP_LOCAL = struct.Struct("<4s5H3L2H")       # a zip member's local file header


def read_npz(path, names):
    """
    The named arrays of an .npz, each read from the file straight into its
    own memory.

    np.load's way through an archive costs more than the bytes it moves: a
    CRC over every member, read in 256 KB pieces and copied twice on the way
    into the array. Measured on the live run, the CRC alone was a tenth of
    the main thread's time. np.savez stores its members uncompressed, so an
    array's bytes lie contiguous in the file and one read puts them in place.
    Anything else - a compressed member, a header this does not parse - goes
    through np.load. Truncation is still caught: a member must be exactly as
    long as its own header says.
    """
    out = {}
    with open(path, "rb", buffering=0) as f:
        members = {zi.filename[:-4]: zi for zi in zipfile.ZipFile(f).infolist()
                   if zi.filename.endswith(".npy")}
        for k in names:
            zi = members.get(k)
            if zi is None:
                continue
            a = _read_stored(f, zi) if zi.compress_type == zipfile.ZIP_STORED else None
            if a is None:
                with np.load(path) as z:
                    a = z[k]
            out[k] = a
    return out


def _read_stored(f, zi):
    f.seek(zi.header_offset)
    sig, *_, n_name, n_extra = _ZIP_LOCAL.unpack(_read_exactly(f, _ZIP_LOCAL.size))
    if sig != b"PK\x03\x04":
        raise ValueError(f"{zi.filename}: no local header where the directory says")
    start = zi.header_offset + _ZIP_LOCAL.size + n_name + n_extra
    f.seek(start)
    version = npy.read_magic(f)
    if version not in ((1, 0), (2, 0)):
        return None
    shape, fortran, dtype = (npy.read_array_header_1_0(f) if version == (1, 0)
                             else npy.read_array_header_2_0(f))
    if dtype.hasobject:
        return None
    a = np.empty(shape, dtype=dtype, order="F" if fortran else "C")
    if f.tell() - start + a.nbytes != zi.file_size:
        raise ValueError(f"{zi.filename}: {zi.file_size} bytes in the archive, "
                         f"{f.tell() - start + a.nbytes} by its own header")
    view = memoryview(a.reshape(-1, order="A").view(np.uint8))
    got = 0
    while got < a.nbytes:
        n = f.readinto(view[got:])
        if not n:
            raise ValueError(f"{zi.filename}: ends {a.nbytes - got} bytes early")
        got += n
    return a


def _read_exactly(f, n):
    b = f.read(n)
    while len(b) < n:                          # an unbuffered read may come short
        more = f.read(n - len(b))
        if not more:
            break
        b += more
    return b


class _CachedCast(torch.autograd.Function):
    """One cast's derivative, backed by reusable detached compute storage."""

    @staticmethod
    def forward(ctx, master, data):
        ctx.master_dtype = master.dtype
        return data

    @staticmethod
    def backward(ctx, grad):
        return grad.to(ctx.master_dtype), None


class Tiers:
    """
    Disk, RAM and VRAM, with an LRU between them.

    Disk is the whole pool, one .npy per expert - the same files the weights
    directory already holds, so nothing new is stored. RAM keeps the most
    recently used `ram_capacity` of them as CPU tensors, so an expert wanted
    again soon costs a copy rather than a read. Least recently used means
    exactly that: when the cache is full, the expert untouched longest is the
    one dropped.

    An expert that was trained while resident is marked dirty and written back
    to disk on eviction. Otherwise the file on disk is still the truth.
    """

    def __init__(self, path, d_model, d_ff, ram_capacity=256, device="cpu",
                 read_only=False):
        self.path = path
        self.d_model, self.d_ff = d_model, d_ff
        self.ram = OrderedDict()             # id -> dict of CPU tensors
        self.ram_capacity = ram_capacity
        self.device = device
        # Read-only means exactly that: nothing is ever marked dirty and
        # nothing is ever written. Needed because paging an expert IN marks it
        # dirty regardless of whether anything changed it, so a visualisation
        # reading a live training directory would otherwise write expert files
        # back underneath the run that owns them.
        self.read_only = read_only
        self.dirty = set()
        self.reads = self.hits = self.evictions = self.writebacks = 0

    def _file(self, i):
        return os.path.join(self.path, "e%05d.npz" % i)

    def _from_disk(self, i):
        """
        One expert, complete: its weights and its optimiser moments.

        The moments live in the same file as the weights because they belong
        to the same thing. Adam's history for an expert is as much a part of
        that expert as its weights are - an expert paged out and back without
        its moments resumes with a stranger's momentum, and one written to disk
        without them cannot be resumed at all. Keeping them together makes the
        file the whole of what that expert is, which is what the weights
        directory claims about itself.
        """
        z = read_npz(self._file(i), _WEIGHTS + _MOMENTS)
        # every array was read into memory of its own, so the tensors take it
        # over rather than copying it
        out = {k: torch.from_numpy(z[k]) for k in _WEIGHTS}
        for k in _MOMENTS:
            if k in z:
                # bf16 if this file has been written since the moments were
                # narrowed, fp32 if it has not; unpack_bf16 reads both
                out[k] = unpack_bf16(z[k])
        return out

    def fetch(self, i, count=True):
        """The expert's CPU tensors, from RAM if it is there.

        `count=False` for a fetch that does not put the expert on the card -
        its moments, fetched for a step - so the hit rate stays a fact about
        loads."""
        if i in self.ram:
            self.hits += count
            self.ram.move_to_end(i)
            return self.ram[i]
        self.reads += count
        e = self._from_disk(i)
        self.ram[i] = e
        self.ram.move_to_end(i)
        self._trim()
        return e

    def put(self, i, tensors, dirty=True):
        """Hand an expert back after it has been resident."""
        self.ram[i] = tensors
        self.ram.move_to_end(i)
        if dirty and not self.read_only:
            self.dirty.add(i)
        self._trim()

    def _trim(self):
        while len(self.ram) > self.ram_capacity:
            j, ent = self.ram.popitem(last=False)      # least recently used
            self.evictions += 1
            if j in self.dirty:
                self._to_disk(j, ent)
                self.dirty.discard(j)

    def _to_disk(self, i, ent):
        if self.read_only:
            raise RuntimeError("read-only pool tried to write e%05d.npz" % i)
        # Weights fp32, Adam's moments bf16. The moments are two thirds of the
        # file and cost a thousandth of their own magnitude to narrow, because
        # what they need is exponent range and bf16 keeps all of fp32's. The
        # weights cannot go with them - see minagi/precision.py, which has the
        # measurement for both.
        arrays = {k: (pack_bf16(v) if is_moment(k) else v.to(torch.float32).numpy())
                  for k, v in ent.items() if torch.is_tensor(v)}
        tmp = self._file(i) + ".tmp.npz"
        np.savez(tmp, **arrays)
        os.replace(tmp, self._file(i))
        self.writebacks += 1

    def flush(self):
        if self.read_only:
            return
        for i in list(self.dirty):
            self._to_disk(i, self.ram[i])
        self.dirty.clear()

    def report(self):
        tot = self.reads + self.hits
        return {"ram_held": len(self.ram), "ram_capacity": self.ram_capacity,
                "disk_reads": self.reads, "ram_hits": self.hits,
                "hit_rate": self.hits / max(tot, 1),
                "evictions": self.evictions, "writebacks": self.writebacks}


class PagedPool(nn.Module):
    """
    A pool whose VRAM cost is set by the card's slots, not by its size.

    Presents enough of SharedPool's surface that PooledMLP routes into it
    unchanged: `gate`, `n_experts()`, `stacked()`, and the usage buffers. What
    differs is that only `resident` experts exist as CUDA parameters at any
    moment, and `stacked()` returns those.

    Routing indices are into the RESIDENT set, so a call site picking expert 3
    means the third of the experts currently held, not the third on disk.
    `self.slots` maps back. Which experts are held is decided by causal admit().
    """

    def __init__(self, path, d_model, d_ff, n_experts, resident=16,
                 ram_capacity=256, device="cpu", max_experts=1_000_000,
                 read_only=False):
        super().__init__()
        self.path = path
        self.d_model, self.d_ff = d_model, d_ff
        self.resident = min(resident, n_experts)
        self.max_experts = max_experts
        self._n = n_experts
        self.read_only = read_only
        self.tiers = Tiers(path, d_model, d_ff, ram_capacity, device,
                           read_only=read_only)

        # the VRAM slots - fixed in number, their contents swapped
        self.w1 = nn.Parameter(torch.zeros(self.resident, d_ff, d_model))
        self.w3 = nn.Parameter(torch.zeros(self.resident, d_ff, d_model))
        self.w2 = nn.Parameter(torch.zeros(self.resident, d_model, d_ff))
        self.slots = [-1] * self.resident          # slot -> expert id
        # What each slot's contents were measured against when its expert was
        # loaded: the slot tensors' version counters and the optimiser steps
        # seen. While both are unchanged the expert is exactly its copy in RAM
        # or on disk, so parking it copies nothing back and writes nothing.
        self._loaded_at = [None] * self.resident
        self._stepped = 0
        self._hooked = set()
        # Whether each slot's optimiser moments are its expert's own. An
        # arriving expert brings its weights only, so its slot's moments are
        # still whatever the slot held before, until _own_moments fills them
        # ahead of a step. `_own_in` is the moment tensors that was true of -
        # replaced ones, as a restore replaces them, are nobody's.
        self._own = [False] * self.resident
        self._own_in = None

        # gates are one float per expert, so the whole pool's worth is nothing
        self.gate = nn.Parameter(torch.ones(n_experts))
        for name in ("use", "age", "born", "gate_seen"):
            self.register_buffer(name, torch.zeros(n_experts),
                                 persistent=False)

        self.grow_events = []
        self.pressure = 0.0
        self.want_k = 0.0
        self.swaps = 0
        self._opt = None
        self._sites = []          # the call sites routing into this pool
        # THE PRUNE CLOCK: the text count at which each expert was last
        # admitted. An expert nothing admits is deleted once this falls far
        # enough behind - see prune() and dying(). A text begins at a forward
        # from position 0; the forwards that continue it through the attention
        # cache admit their own experts but do not move the clock, so writing
        # a reply one character at a time does not age the pool.
        self.register_buffer("last_seen", torch.zeros(n_experts),
                             persistent=False)
        # texts begun so far - the unit last_seen is counted in
        self.segments = 0
        # whether an expert has ever been on the card, and how many times it
        # has been brought there, over the model's whole life. Record only:
        # neither takes part in choosing.
        self.register_buffer("ever", torch.zeros(n_experts, dtype=torch.bool),
                             persistent=False)
        self.register_buffer("admits", torch.zeros(n_experts),
                             persistent=False)
        # An expert's NAME is its uid, not its position. Position changes
        # every time something is pruned; a uid never does. That is what keeps
        # a file where it is across a prune, and what lets a record written
        # a million characters ago still mean the same expert.
        self.register_buffer("uid", torch.arange(n_experts, dtype=torch.long),
                             persistent=False)
        self.next_uid = int(n_experts)
        # A newborn is safe from pruning for its first `trial` steps; `now` is
        # the training step, set by the reader. Without it nothing is ever on
        # trial, which is the safe default.
        self.trial = 0
        self.now = 0
        # share of the survival window unaddressed that counts as dying
        self.dying_at = 0.75
        # the union of the current sequences' causal working sets
        self._admitted = set()
        # Detached compute data only; each training use gets its own FP32
        # gradient edge. Never overwrite a generation held by a backward graph.
        self._infer = None
        self.cache_training_weights = True
        self._mask = None
        # the slots as an index on a device - see _slot_rows()
        self._slot_idx = ((), {})
        self.loads = 0            # experts brought onto the card, ever
        # HOW MUCH EACH EXPERT HAS BEEN USED LATELY: its share of recent
        # training forwards that admitted it, a running average over
        # `usage_steps` of them. Every forward that trains updates it.
        self.register_buffer("recent", torch.zeros(n_experts),
                             persistent=False)
        self.usage_steps = 1000.0        # how far back `recent` looks
        self._trains = False             # whether this forward trains
        # THE BALANCE TERM - a term in the loss, `balance` its weight, that
        # charges the router for the probability it puts on each expert in
        # proportion to that expert's share of recent admissions. Computed by
        # the call site in the first pass of a forward that trains; read by the
        # model's pool_balance(). See note_balance.
        self.balance = 0.0
        self._balance = None
        # A sequence fills its working set in prefix order on the first
        # recurrent pass. Its set then stays fixed until the text resets.
        # Masks distinguish physical residency from causal availability:
        # loading a later position's expert never exposes it to an earlier one.
        self._batch_size = 1
        self._sequence_sets = [set()]
        self._prefix_mask = None
        self._offset = 0
        self._rng_seed = None
        self._noise_pages = {}
        self.freeze_admission = False       # replication's fixed-pool control
        self.select_temperature = 0.0

    def _f(self, i):
        """Position in the pool's arrays -> the id its file is named by."""
        return int(self.uid[int(i)])

    def dying(self):
        """
        How close each expert is to being deleted, as a share of the window.

        0.0 means something asked for it just now. 1.0 means it has gone
        exactly as long unaddressed as `survival`, which is the line prune
        deletes on. Above 1.0 it is only alive because prune has not run yet.

        DEAD MEANS UNADDRESSED, and this is the one definition prune acts on.
        It is deliberately not a gate question. The gate says how loudly an
        expert speaks when chosen, and the smallest gates belong to the
        BUSIEST experts - a sink that is picked constantly and contributes
        little per character reads as dead on any gate test, while a rarely
        chosen expert with a large gate reads as alive. Time since anything
        last wanted it is the quantity that actually predicts being wanted
        again.

        A newborn inside its trial returns 0. It has not failed to be chosen;
        it has not finished being offered.
        """
        n = self._n
        z = torch.zeros(n, device=self.gate.device)
        survival = float(getattr(self, "trial", 0) or 0)
        step = float(getattr(self, "now", 0) or 0)
        if n == 0 or survival <= 0 or step <= 0:
            return z
        # the same steps-to-segments conversion prune uses, for the same reason
        per_step = self.segments / max(step, 1.0)
        window = survival * per_step
        if window <= 0:
            return z
        # An expert cannot have gone unaddressed for longer than it has
        # existed. `last_seen` starts at 0 for a newborn, and read as "how
        # long since anything wanted it" that 0 would mean "since the
        # beginning of the run". Clamping to the expert's own age is what stops
        # a newborn scoring past the prune line from the moment it exists.
        born_seg = (step - self.born[:n].to(z.dtype)).clamp_min(0.0) * per_step
        idle = (self.segments - self.last_seen[:n].to(z.dtype)).clamp_min(0.0)
        frac = torch.minimum(idle, born_seg) / window
        young = (step - self.born[:n].to(z.dtype)) < survival
        return torch.where(young, z, frac)


    def _carry(self, opt, old_p, new_p, idx=None, grow=0):
        """
        Move Adam's moments from a replaced parameter onto its replacement.

        Growing or pruning the pool builds a NEW gate tensor and NEW router
        rows. Without this the optimiser would drop the state attached to the
        old ones, and the machinery that decides WHICH expert to use would
        restart its moment estimates at every growth decision. Pruning slices
        rows out, growth appends zeroed ones, and the moments follow the same
        shape either way.
        """
        if opt is None:
            return
        st = opt.state.pop(old_p, None)
        if not st:
            return
        out = {}
        for k, v in st.items():
            if torch.is_tensor(v) and v.dim() and v.shape[0] == old_p.shape[0]:
                if idx is not None:
                    v = v[idx].clone()
                elif grow:
                    pad = torch.zeros((grow,) + tuple(v.shape[1:]),
                                      dtype=v.dtype, device=v.device)
                    v = torch.cat([v, pad])
            out[k] = v
        opt.state[new_p] = out

    # -- the surface PooledMLP expects ------------------------------------
    def n_experts(self):
        return self._n

    def n_routable(self):
        """A token chooses only between the experts in VRAM."""
        return self.resident

    def router_rows(self):
        """One row per expert on disk; growth adds rows."""
        return self._n

    def _slot_rows(self, device):
        """
        The expert each slot holds, as an index on `device`; an empty slot
        reads row 0. Every row of every forward asks for it - for the
        router's rows, the gates, the usage count - and on a GPU a tensor
        built from a list is a copy the host stops and waits for, so it is
        built again only when the slots change.
        """
        key = tuple(self.slots)
        if self._slot_idx[0] != key:
            self._slot_idx = (key, {})
        got = self._slot_idx[1].get(device)
        if got is None:
            got = self._slot_idx[1][device] = torch.tensor(
                [max(s, 0) for s in key], device=device)
        return got

    def resident_rows(self):
        """Which router rows the resident experts own, in slot order."""
        return self._slot_rows(self.gate.device)

    def n_resident(self):
        return self.resident

    def routable_gate(self):
        """Gates of the resident experts, in slot order."""
        return self.gate[self._slot_rows(self.gate.device)]

    def note_use(self, hit):
        """Routing counts arrive per SLOT; usage is kept per EXPERT."""
        self.use.index_add_(0, self._slot_rows(self.use.device),
                            hit.to(self.use.dtype))
        self.age += 1

    def n_params(self):
        per = 3 * self.d_model * self.d_ff
        return self._n * per + self.gate.numel()

    def disk_bytes(self, extra=0):
        """
        What the pool costs on disk, and what `extra` more experts would cost.

        EIGHT bytes per parameter, which is what the files actually hold: one
        fp32 weight (4) plus Adam's two moments as bf16 (2 + 2). At d_model 512
        and d_ff 2048 that is 25.2 MB an expert.

        `growth.max_disk_gb` is enforced against this number, so it has to
        match the files rather than the widest possible layout - an estimate
        that assumed fp32 moments would run 1.5x over and stop growth well
        short of the budget it was given.
        """
        per = 3 * self.d_model * self.d_ff * 8
        return (self._n + int(extra)) * per

    def vram_params(self):
        return (3 * self.resident * self.d_model * self.d_ff
                + self.gate.numel())


    def stacked(self):
        return [(self.w1, self.w3, self.w2)]

    def invalidate(self):
        self._infer = None

    def _apply(self, fn, recurse=True):
        self.invalidate()  # release old-device storage when moving the module
        return super()._apply(fn, recurse=recurse)

    # -- causal admission -------------------------------------------------

    def begin_forward(self, explore=False, *, pos_offset=0, batch_size=1):
        """Start a frame while retaining each sequence's prefix working set."""
        if batch_size != self._batch_size:
            raise ValueError("cached paging cannot change the sequence batch size")
        self._prefix_mask = None
        self._offset = pos_offset
        self._balance = None
        self._trains = bool(explore)
        if self._trains:
            self.recent[:self._n] *= 1.0 - 1.0 / self.usage_steps
            # The running share counts admissions per training forward,
            # including experts carried from its preceding cached frame.
            if self._admitted:
                self.recent[list(self._admitted)] += 1.0 / self.usage_steps

    def begin_text(self, explore=False, *, batch_size=1):
        """Reset causal state at a context boundary, never at a chunk boundary."""
        if batch_size < 1 or batch_size > self.resident:
            raise ValueError("paging needs at least one resident slot per sequence")
        if self.freeze_admission:
            if batch_size != self._batch_size:
                raise ValueError("the fixed-pool control cannot change batch size")
        else:
            self._admitted = set()
            self._sequence_sets = [set() for _ in range(batch_size)]
        self._batch_size = batch_size
        self._rng_seed = (int(torch.randint(0, 2**62, ()).item())
                          if self.select_temperature > 0 else None)
        self._noise_pages = {}
        self.begin_forward(explore, batch_size=batch_size)
        self.segments += 1

    def sequence_capacity(self):
        """Independent sequence budgets keep batching from changing other inputs."""
        return self.resident // self._batch_size

    def routing_noise(self, positions, *, site, depth, width, admission=False):
        """Gumbel draws keyed by sequence, absolute position, site and depth.

        A global RNG advanced in depth-major order gives earlier tokens different
        draws when a suffix or a chunk is added. Local generators make the random
        draw independent of execution order. Columns are expert IDs, not slots.
        """
        if self._rng_seed is None:
            raise RuntimeError("stochastic paging needs begin_text before routing")
        # Fixed pages amortize RNG kernels during tokenwise decoding. Keep
        # only the last page per site/depth/sequence, bounding the cache even
        # for long contexts. Regenerating an older page gives the same draws.
        noise = []
        key = lambda bp: (bp[0], bp[1] // 128)
        for (batch, page), group in itertools.groupby(positions, key):
            cache_key = (batch, site, depth, width, admission)
            cached = self._noise_pages.get(cache_key)
            if cached is None or cached[0] != page:
                identity = (self._rng_seed, batch, page, site, depth, admission)
                digest = hashlib.blake2b(repr(identity).encode(), digest_size=8).digest()
                seed = int.from_bytes(digest, "little") % (2**63 - 1)
                gen = torch.Generator(device=self.gate.device).manual_seed(seed)
                u = torch.rand(128, width, device=self.gate.device, generator=gen)
                values = -torch.log(-torch.log(u.clamp(1e-7, 1 - 1e-7)))
                cached = self._noise_pages[cache_key] = (page, values)
            within_page = torch.tensor([pos % 128 for _, pos in group],
                                       device=self.gate.device)
            noise.append(cached[1][within_page])
        return torch.cat(noise)

    def usage_share(self):
        """Each expert's share of recent admissions, summing to 1 - even when
        nothing has been used yet."""
        n = self._n
        r = self.recent[:n].float()
        s = float(r.sum())
        if s <= 0:
            return torch.full((n,), 1.0 / max(n, 1), device=r.device)
        return r / s

    def note_balance(self, term):
        """
        The balance term a call site computed for this forward, weighted.

        It is the Switch Transformer's balancing term with one change: each
        expert is charged by its share of the last `usage_steps` training
        forwards' admissions rather than of this forward's, because a forward
        is one text and a text should be free to want few experts - what has
        to be even is use across texts. Probability on a busy expert costs
        more than on an idle one, so the router's rows move toward the idle,
        and since it is the router itself that changes, reading and writing
        choose the same way. Its weight sets how hard it leans against the
        language-model gradient on experts in use; an expert no text admits
        gets no other gradient, so Adam moves its row at the usual pace
        whatever the weight.
        """
        self._balance = term if self._balance is None else self._balance + term

    def balance_term(self):
        """This forward's balance term, or None when it has none."""
        return self._balance

    def admitting(self):
        return (not self.freeze_admission and any(
            len(chosen) < self.sequence_capacity() for chosen in self._sequence_sets))

    def admit(self, requests):
        """Build availability from ordered first-pass requests [batch, time, k].

        Each sequence has a fixed share of the resident slots. It accepts new
        experts in request order until that share is full. All physical loads
        happen before expert computation, but a position's mask includes only
        experts accepted at or before that position. No selected slot is evicted
        until a new text, so backward and Adam still see the same expert weights.
        """
        if requests.ndim != 3 or requests.shape[0] != self._batch_size:
            raise ValueError("admission requests must be [batch, time, k]")
        B, T, _ = requests.shape
        if not self.admitting():
            key = (tuple(self.slots), tuple(frozenset(s) for s in self._sequence_sets))
            if self._mask is None or self._mask[0] != key:
                mask = torch.tensor([[e in chosen for e in self.slots]
                                     for chosen in self._sequence_sets],
                                    device=self.gate.device)
                self._mask = (key, mask)
            self._prefix_mask = self._mask[1].unsqueeze(1).expand(B, T, -1)
            return self._prefix_mask
        births, new = [], []
        all_chosen = set(self._admitted)
        for chosen, sequence in zip(self._sequence_sets, requests.tolist()):
            first = {e: -1 for e in chosen}
            if not self.freeze_admission:
                for position, wanted in enumerate(sequence):
                    if len(chosen) >= self.sequence_capacity():
                        break
                    for e in wanted:
                        if e not in chosen:
                            chosen.add(e)
                            first[e] = position
                            if e not in all_chosen:
                                all_chosen.add(e)
                                new.append(e)
                        if len(chosen) >= self.sequence_capacity():
                            break
            births.append(first)
        if new:
            self._place(new)
        introduced = torch.tensor(
            [[first.get(e, T) if e >= 0 else T for e in self.slots]
             for first in births], device=self.gate.device)
        position = torch.arange(T, device=self.gate.device).view(1, T, 1)
        self._prefix_mask = position >= introduced.unsqueeze(1)
        return self._prefix_mask

    def _place(self, new):
        """Put newly admitted experts on the card. Returns how many loaded."""
        adm = self._admitted
        adm |= set(new)
        here = {e: s for s, e in enumerate(self.slots) if e >= 0}
        # a newcomer takes a slot holding nothing this forward admitted: an
        # empty one first, then the one whose expert was admitted longest ago -
        # the least recently used, which is the one least likely to be wanted
        # back
        victims = [s for s, e in enumerate(self.slots) if e < 0 or e not in adm]
        if victims:
            seen = self.last_seen.tolist()       # one read-back, not one per slot
            victims.sort(key=lambda s: (self.slots[s] >= 0,
                                        seen[self.slots[s]]
                                        if self.slots[s] >= 0 else -1.0))
        plan = list(self.slots)
        for e in new:
            if e not in here:
                plan[victims.pop(0)] = e
        loads = sum(1 for e in plan if e >= 0 and e not in here)
        if plan != self.slots:
            for e in plan:
                if e >= 0 and e not in here:
                    self.admits[e] += 1
            self._rearrange(plan, here)
            self.swaps += 1
        for e in new:
            self.ever[e] = True
        # THE PRUNE CLOCK: whatever admits an expert resets it
        self.last_seen[torch.tensor(new, device=self.last_seen.device)] = \
            float(self.segments)
        if self._trains:
            self.recent[new] += 1.0 / self.usage_steps
        self.loads += loads
        return loads

    def inference_weights(self, dtype):
        """Detached casts, reused until an update, load or storage change.

        Ordinary in-place edits/load_state_dict advance version counters.
        Paging writes through .data and explicitly invalidates instead. Other
        external .data writes must call invalidate(), just as for inference.
        All slab identities/storage properties are checked, including when
        load_state_dict(assign=True) replaces a parameter.
        """
        weights = (self.w1, self.w3, self.w2)
        key = (dtype, self._stepped, self.swaps, tuple(self.slots),
               tuple((id(t), t.data_ptr(), t._version, t.device, t.dtype,
                      tuple(t.shape), t.stride()) for t in weights))
        if self._infer is None or self._infer[0] != key:
            # An inference-mode-created tensor cannot be saved by autograd.
            # Make normal detached tensors so a later training call can reuse
            # this generation, including inside activation checkpointing.
            with torch.inference_mode(False), torch.no_grad():
                self._infer = (key, tuple(t.detach().to(dtype) for t in weights))
        return self._infer[1]

    def training_weights(self, dtype):
        """Reuse cast data, never a differentiable cast shared across uses.

        A shared .to(dtype) graph would add gradients in the compute dtype
        before converting to FP32. Separate edges preserve the original
        cast-per-dispatch accumulation, also with activation checkpointing.
        The switch provides an uncached reference path for A/B verification.
        """
        weights = (self.w1, self.w3, self.w2)
        if not self.cache_training_weights or all(t.dtype == dtype for t in weights):
            return weights
        data = self.inference_weights(dtype)
        return tuple(_CachedCast.apply(t, d) if t.dtype != dtype else t
                     for t, d in zip(weights, data))

    def admitted_mask(self, positions=None):
        """Per-position availability; without positions, physical membership."""
        if positions is not None:
            if self._prefix_mask is None:
                raise RuntimeError("the first recurrent pass must prepare admission")
            return self._prefix_mask.flatten(0, 1)[positions]
        return torch.tensor([e in self._admitted for e in self.slots],
                            device=self.gate.device)

    @torch.no_grad()
    def _rearrange(self, plan, here):
        """
        Put the card into the state `plan` describes: park what is leaving,
        fetch what is arriving, and leave everything else where it is.

        An arriving expert brings its weights and nothing else. Its moments
        belong to it as much as its weights do, but only an optimiser step
        reads them, so they come to the card just before one - _own_moments -
        and only for the experts on the card at that moment. A reply loads
        experts at almost every character and steps none of them; carrying
        their moments as well would double what each load copies, for nothing.

        An expert that nothing has trained since it was loaded is still exactly
        its copy in RAM or on disk, so parking it copies nothing back - a
        reply written one character at a time loads experts constantly and
        changes none of them, and writing them back would put every one of
        those loads on the disk a second time for nothing. One that was
        stepped goes back with its weights and its moments.
        """
        st = (self._opt.state if self._opt is not None else {})
        tensors = ((self.w1, "w1"), (self.w3, "w3"), (self.w2, "w2"))
        old = list(self.slots)
        now = self._slot_state()

        for e in old:
            if e >= 0 and e not in plan:
                s = here[e]
                if self._loaded_at[s] == now:
                    continue                  # unchanged since it was loaded
                self.tiers.put(self._f(e), self._entry(s, e, st), dirty=True)

        for s, e in enumerate(plan):
            if e < 0 or old[s] == e:
                continue
            src = self.tiers.fetch(self._f(e))
            for p, nm in tensors:
                p.data[s].copy_(src[nm])
            self._own[s] = False
        now = self._slot_state()
        for s, e in enumerate(plan):
            if e >= 0 and old[s] != e:
                self._loaded_at[s] = now
        self.slots = list(plan)
        # Loads write through .data, which does not advance the slots'
        # version counters - so the cast copy is dropped here explicitly
        self.invalidate()

    def _entry(self, s, e, st):
        """
        Slot s as expert e's entry in RAM: its weights, and its moments - the
        optimiser's when they are its own there, which they are for every
        expert a step has touched, and otherwise the ones its entry already
        holds. An entry replaces the old one whole, so it never leaves them out.
        """
        tensors = ((self.w1, "w1"), (self.w3, "w3"), (self.w2, "w2"))
        ent = {nm: p.data[s].detach().to("cpu").clone() for p, nm in tensors}
        if self._own[s]:
            for p, nm in tensors:
                o = st.get(p)
                if o and "exp_avg" in o:
                    ent[nm + "_m"] = o["exp_avg"][s].to("cpu").clone()
                    ent[nm + "_v"] = o["exp_avg_sq"][s].to("cpu").clone()
        else:
            ent.update({k: v for k, v in
                        self.tiers.fetch(self._f(e), count=False).items()
                        if is_moment(k)})
        return ent

    def _moment_state(self):
        """The optimiser's state for the three slot tensors - or None with no
        optimiser, or before its first step has created the moments."""
        if self._opt is None:
            return None
        sts = [self._opt.state.get(p) for p in (self.w1, self.w3, self.w2)]
        return sts if all(o and "exp_avg" in o for o in sts) else None

    @torch.no_grad()
    def _own_moments(self, *_):
        """
        Before every optimiser step: each expert on the card holds its own
        moments, and one that does not yet gets them now.

        AdamW steps every slot, used or not, so every expert on the card at a
        step is stepped, and it must be stepped with its own history. An
        arriving expert brought only its weights, so its moments are copied in
        here from its entry in RAM or on disk - the latest it has, because
        nothing has stepped it since it was loaded. From then until another
        expert takes the slot, the step keeps them its own.

        Moments filled into tensors the optimiser has since replaced are
        nobody's: a restore from optim.npz writes the slot-indexed moments of
        whichever experts sat in the slots when it was saved, so after one
        every slot is filled again. Before an optimiser's first step there
        are no moment tensors to fill; Adam creates them, at zero, in it.
        """
        sts = self._moment_state()
        if sts is None:
            return
        if self._own_in is None or any(r() is not o["exp_avg"]
                                       for r, o in zip(self._own_in, sts)):
            self._own = [False] * self.resident
        names = ("w1", "w3", "w2")
        for s, e in enumerate(self.slots):
            if e < 0 or self._own[s]:
                continue
            src = self.tiers.fetch(self._f(e), count=False)
            for nm, o in zip(names, sts):
                if nm + "_m" in src:
                    o["exp_avg"][s].copy_(src[nm + "_m"])
                    o["exp_avg_sq"][s].copy_(src[nm + "_v"])
                else:
                    o["exp_avg"][s].zero_()
                    o["exp_avg_sq"][s].zero_()
            self._own[s] = True
        self._own_in = [weakref.ref(o["exp_avg"]) for o in sts]

    def _slot_state(self):
        """
        What changes when the slots are trained: the slot tensors' version
        counters, which every optimiser step on them moves, and the optimiser
        steps seen - the second in case an optimiser writes the slots without
        moving the counters. Loading writes through .data, which moves
        neither, so a slot's state at load is what it is compared against.
        """
        return (self.w1._version, self.w3._version, self.w2._version,
                self._stepped)

    def _count_step(self, *_):
        self._stepped += 1
        self.invalidate()
        # whatever each slot's moments were going in - its own, or the zeros
        # an optimiser's first step starts from - they are its expert's now
        sts = self._moment_state()
        if sts is not None:
            self._own = [e >= 0 for e in self.slots]
            self._own_in = [weakref.ref(o["exp_avg"]) for o in sts]

    # -- growing and pruning ----------------------------------------------
    @torch.no_grad()
    def add_experts(self, k, seed_from=None, device=None, step=0,
                    birth_gate=0.001, make=None, recombine=16):
        """
        Write k new experts to disk and make them part of the pool.

        They are not loaded. A new expert only reaches VRAM if a forward admits
        it. Growth is therefore cheap in a way it never was while every expert
        had to be resident: adding a hundred experts costs a hundred files and
        not one megabyte of card.

        HOW ONE IS BUILT is RECOMBINATION, by default from `recombine` other
        experts. A hidden unit is the triple (w1[u], w3[u], w2[:, u]) and units
        are interchangeable, so a child assembled from whole units taken from
        different parents keeps every unit's learned feature intact while
        computing a function nothing in the pool computes.

        The two obvious alternatives both fail, in opposite directions. A clone
        plus noise is not novel - its output sits at ~0.98 cosine to its parent
        where two established experts sit at ~0.04 - so it only double-counts
        what the pool already has. A fresh random expert is novel but computes
        nothing worth routing to. What works is novelty built from TRAINED
        parts. The comparison is in `runs/results/birth_schemes.json`.

        A new expert is born at `birth_gate`, a small starting scale rather
        than a verdict: prune reads staleness, not the gate, so what keeps a
        newborn alive is its trial window and then being asked for.

        ITS ROUTER ROW IS ITS PARENTS'. Every router gains a row per newborn,
        and that row is the average of the rows of the experts its units came
        from, weighted by how many units each gave. A row is what decides
        whether any text ever asks for the expert, and a random one is asked
        for by chance or not at all: this one is asked for where its parents
        are, which is where its units already know what to do.
        """
        dev = self.gate.device
        src = None
        if seed_from is not None and 0 <= int(seed_from) < self._n:
            src = self.tiers.fetch(self._f(int(seed_from)))

        # The sources recombination draws from: a sample of the pool, with the
        # expert the caller asked to split kept among them. Fetched once and
        # reused for every child in this batch.
        srcs = None
        if make is None and recombine and recombine >= 2 and self._n >= 2:
            m = min(int(recombine), self._n)
            pick = torch.randperm(self._n)[:m].tolist()
            if (seed_from is not None and 0 <= int(seed_from) < self._n
                    and int(seed_from) not in pick):
                pick[0] = int(seed_from)
            got = [self.tiers.fetch(self._f(int(i))) for i in pick]
            srcs = tuple(torch.stack([g[nm].float() for g in got])
                         for nm in ("w1", "w3", "w2"))
        new_uids = []
        lineage = []              # per child: (parent positions, unit shares)
        for _ in range(k):
            i = self._n
            nu = self.next_uid
            self.next_uid += 1
            if make is not None:
                # a birth scheme supplied by the caller, for comparing what a
                # newborn should BE against the default. See
                # tools/birth_probe.py.
                ent = {nm: t.detach().clone()
                       for nm, t in zip(("w1", "w3", "w2"), make(i, src))}
                lineage.append((torch.tensor([int(seed_from)]), torch.ones(1))
                               if src is not None else None)
            elif srcs is not None:
                # RECOMBINATION. A hidden unit is (w1[u], w3[u], w2[:, u]) and
                # units are interchangeable, so taking whole units from
                # different experts keeps every unit's learned feature intact
                # while composing a function nothing in the pool computes.
                who = torch.randint(0, srcs[0].shape[0], (self.d_ff,))
                unit = torch.arange(self.d_ff)
                ent = {"w1": srcs[0][who, unit].clone(),
                       "w3": srcs[1][who, unit].clone(),
                       "w2": srcs[2][who, :, unit].t().contiguous().clone()}
                share = torch.bincount(who, minlength=srcs[0].shape[0]).float()
                lineage.append((torch.tensor(pick), share / share.sum()))
            elif src is not None:
                ent = {nm: (src[nm] + 0.02 * torch.randn_like(src[nm])).clone()
                       for nm in ("w1", "w3", "w2")}
                lineage.append((torch.tensor([int(seed_from)]),
                                torch.ones(1)))
            else:
                ent = {"w1": torch.randn(self.d_ff, self.d_model) * 0.02,
                       "w3": torch.randn(self.d_ff, self.d_model) * 0.02,
                       "w2": torch.randn(self.d_model, self.d_ff) * 0.02}
                lineage.append(None)
            self.tiers.put(nu, ent, dirty=True)
            new_uids.append(nu)
            self._n += 1
        self.tiers.flush()
        self.uid = torch.cat([self.uid,
                              torch.tensor(new_uids, dtype=torch.long,
                                           device=self.uid.device)])

        def grow_vec(t, fill=0.0):
            return torch.cat([t, torch.full((k,), float(fill),
                                            device=t.device, dtype=t.dtype)])
        opt = getattr(self, "_opt", None)
        old_gate = self.gate
        self.gate = nn.Parameter(grow_vec(self.gate.data, birth_gate))
        self._carry(opt, old_gate, self.gate, grow=k)
        self.last_seen = grow_vec(self.last_seen, 0.0)
        self.use = grow_vec(self.use)
        self.age = grow_vec(self.age)
        self.born = grow_vec(self.born, step)
        self.gate_seen = grow_vec(self.gate_seen)
        self.ever = torch.cat([self.ever, torch.zeros(k, dtype=torch.bool,
                                                      device=self.ever.device)])
        self.admits = grow_vec(self.admits, 0.0)
        # a newborn has been used by nothing, so the balance term pulls its row
        # up from the first training forward after its birth
        self.recent = grow_vec(self.recent, 0.0)

        # the routers keep one row per expert, so they grow too - each new
        # row the unit-weighted average of its parents' rows (see above)
        for site in self._sites:
            rw = site.router.weight.data
            if rw.shape[0] < self._n:
                rows = []
                for spec in lineage:
                    if spec is None:
                        rows.append(torch.randn(self.d_model, device=rw.device,
                                                dtype=rw.dtype) * 0.01)
                    else:
                        who_, share = spec
                        rows.append(share.to(rw.device, rw.dtype)
                                    @ rw[who_.to(rw.device)])
                extra = torch.stack(rows)
                old_r = site.router.weight
                site.router = nn.Linear(self.d_model, self._n,
                                        bias=False).to(rw.device)
                site.router.weight.data = torch.cat([rw, extra])
                self._carry(opt, old_r, site.router.weight,
                            grow=self._n - rw.shape[0])
        return self._n

    @torch.no_grad()
    def prune(self, step, survival=8600, protect=0):
        """
        Delete experts that have atrophied. Their files go with them.

        USE IT OR LOSE IT, and USE IS THE ONLY TEST. An expert is removed when
        it has not been admitted to the card once in THE LAST `survival` steps,
        and it is never touched at all in its first `survival` steps.

          stale    a trailing window, not a one-off check at birth: an expert
                   that worked for 200,000 steps and has since gone quiet dies
                   too. Age earns nothing permanent. Not "its gate is low",
                   not "its gate stopped rising" - it was not chosen.

        THERE IS NO GATE TERM HERE, deliberately. The gate is not merely
        uninformative about whether an expert will be wanted again, it is
        anti-predictive: the smallest gates belong to the busiest experts. One
        that behaves as a sink - chosen constantly, contributing little per
        character - reads as dead on a gate test, while a high-gate expert
        nothing has asked for in hundreds of thousands of texts reads as
        alive. A gate test would delete the first and spare the second.

        The growth brake reads the SAME staleness, through dying(): an expert
        is dying once it has gone `dying_at` of the way to this window without
        being addressed. One definition, two thresholds - one that stops
        growth, one that deletes.

        A newborn is safe for `survival` steps no matter what, so it cannot be
        judged before texts have had the chance to ask for it.

        THE COST OF THIS TRADE is that an expert which is genuinely rare rather
        than dead is deleted, and deletion is permanent. `survival` is the only
        thing holding that, so it should be set wide.
        """
        # `last_seen` counts TEXTS and `survival` is in steps, so the window
        # is converted with the pool's own cumulative ratio rather than a
        # constant - it is self-calibrating and needs nothing stored.
        per_step = self.segments / max(float(step), 1.0)
        window = survival * per_step if per_step > 0 else float("inf")
        keep = []
        for i in range(self._n):
            if i < protect or i in self.slots:
                keep.append(i)                 # never drop what is resident
                continue
            young = (step - float(self.born[i])) < survival
            seen = (self.segments - float(self.last_seen[i])) <= window
            if young or seen:
                keep.append(i)
        if len(keep) == self._n:
            return 0
        if not keep:
            # Everything qualified and nothing was resident to protect it. A
            # pool of zero experts cannot route, so one stays whatever the
            # thresholds say - a rule that never fires in a healthy run and
            # stops a collapsed one from destroying itself. It is the most
            # recently wanted one, not the best-gated one: the gate is no
            # longer what this rule is about, and the last expert anything
            # asked for is the least bad thing to be left holding.
            keep = [int(self.last_seen[:self._n].argmax())]
        gone = self._n - len(keep)
        kept = set(keep)

        # Files are named by uid, so NOTHING is renamed. A prune that
        # renumbered the directory would cost up to n renames for one deletion,
        # and every record ever written by position - expert_history.jsonl
        # included - would silently stop meaning what it said.
        for i in range(self._n):
            if i not in kept:
                u = self._f(i)
                f = self.tiers._file(u)
                if os.path.exists(f):
                    os.remove(f)
                self.tiers.ram.pop(u, None)
                self.tiers.dirty.discard(u)

        idx = torch.tensor(keep, dtype=torch.long, device=self.gate.device)
        opt = getattr(self, "_opt", None)
        old_gate = self.gate
        self.gate = nn.Parameter(self.gate.data[idx].clone())
        self._carry(opt, old_gate, self.gate, idx=idx)
        for nm in ("use", "age", "born", "gate_seen", "last_seen", "ever",
                   "admits", "uid", "recent"):
            setattr(self, nm, getattr(self, nm)[idx].clone())
        # Give every router a NEW parameter rather than reshaping the one it
        # has. Autograd sizes a gradient from the tensor it saved, so shrinking
        # a Parameter in place under a graph that still references it makes the
        # backward return the new number of rows where the graph recorded the
        # old one - which is a crash, and only in a run that prunes. Growth
        # never hit it because add_experts already allocates a fresh Linear.
        for site in self._sites:
            w = site.router.weight
            fresh = nn.Linear(w.shape[1], len(keep), bias=False).to(w.device)
            fresh.weight.data = w.data[idx].clone()
            site.router = fresh
            self._carry(opt, w, fresh.weight, idx=idx)
        remap = {old_i: new_i for new_i, old_i in enumerate(keep)}
        self.slots = [remap.get(s, -1) for s in self.slots]
        self._admitted = {remap[e] for e in self._admitted if e in remap}
        self._sequence_sets = [{remap[e] for e in chosen if e in remap}
                               for chosen in self._sequence_sets]
        self._prefix_mask = None
        self._n = len(keep)
        return gone

    def saturation(self):
        """
        What the growth brakes read. Idle is a STALENESS question - see
        dying(). It is deliberately not measured on routing traffic: the
        load-balancing loss makes routing near-uniform by design, so anything
        counted from traffic describes that objective rather than the pool.
        """
        u = self.use / self.use.sum().clamp_min(1)
        n = max(u.numel(), 1)
        ideal = 1.0 / n
        ent = float(-(u.clamp_min(1e-9) * u.clamp_min(1e-9).log()).sum())
        # Idle is a STALENESS question, not a gate one. See dying().
        d = self.dying()
        return {"experts": self._n,
                "idle": int((d >= self.dying_at).sum()),
                "dying_at": self.dying_at,
                "idle_by_routing": int((u < 0.1 * ideal).sum()),
                "entropy_frac": ent / max(math.log(n), 1e-9),
                "peak_over_ideal": float(u.max()) / ideal if n else 0.0,
                "pressure": float(self.pressure),
                "want_k": float(self.want_k)}

    def telemetry(self):
        """
        Per-expert history, for a checkpoint to carry and a tool to read back.

        None of it is reconstructable afterwards. The usage counters are
        rebuilt every run, and an expert's file records only when the cache
        last wrote it, which is a fact about the cache and not about use.
        """
        return {"gate": [round(float(v), 6) for v in self.gate.data],
                # What the gate was at the last prune check. Carried so a
                # resumed run can still tell a gate that is small but climbing
                # from one that has never moved; this pool's own prune reads
                # staleness instead, so nothing here depends on it.
                "gate_seen": [round(float(v), 6) for v in self.gate_seen],
                "use": [float(v) for v in self.use],
                "admits": [float(v) for v in self.admits],
                "born": [float(v) for v in self.born],
                "last_seen": [float(v) for v in self.last_seen],
                "recent": [round(float(v), 6) for v in self.recent],
                "ever": [bool(v) for v in self.ever],
                "uid": [int(v) for v in self.uid],
                "next_uid": int(self.next_uid),
                "segments": int(self.segments)}

    def load_telemetry(self, t):
        """Put back what a checkpoint carried, ignoring anything resized."""
        if not t:
            return
        for nm in ("use", "admits", "born", "last_seen",
                   "gate_seen", "recent"):
            v = t.get(nm)
            if not v:
                continue
            b = getattr(self, nm)
            k = min(len(v), b.numel())
            b[:k] = torch.tensor(v[:k], dtype=b.dtype, device=b.device)
        ev = t.get("ever")
        if ev:
            k = min(len(ev), self.ever.numel())
            self.ever[:k] = torch.tensor(ev[:k], dtype=torch.bool,
                                         device=self.ever.device)
        u = t.get("uid")
        if u:
            k = min(len(u), self.uid.numel())
            self.uid[:k] = torch.tensor(u[:k], dtype=torch.long,
                                        device=self.uid.device)
        # A directory written before ids existed has files named by position,
        # which is exactly what uid = arange gives, so it loads unchanged.
        self.next_uid = int(t.get("next_uid")
                            or (int(self.uid.max()) + 1 if self.uid.numel()
                                else 0))
        self.segments = int(t.get("segments") or self.segments)

    def attach_optimiser(self, opt):
        """
        The optimiser whose steps the experts on the card take part in: before
        each step every expert on the card gets its own moments
        (_own_moments), after it the step is counted (_count_step). A
        different optimiser holds nobody's moments yet, so every slot is
        filled again before its first step.
        """
        if opt is not self._opt:
            self._own = [False] * self.resident
            self._own_in = None
        self._opt = opt
        if opt is not None and id(opt) not in self._hooked:
            opt.register_step_pre_hook(self._own_moments)
            opt.register_step_post_hook(self._count_step)
            self._hooked.add(id(opt))

    def attach_sites(self, model):
        """
        Remember the call sites, so growth can extend their routers.

        Their rows belong to experts, not to VRAM slots, so a pool that gains
        an expert must give every router a row for it - otherwise the new
        expert is unaddressable by the thing that decides what runs.
        """
        from minagi.pool import PooledMLP
        self._sites = [m for m in model.modules() if isinstance(m, PooledMLP)]
        return len(self._sites)

    def flush(self):
        """Park every resident expert that has changed, then write everything
        dirty to disk. Afterwards each one matches its copy again."""
        st = (self._opt.state if self._opt is not None else {})
        now = self._slot_state()
        for slot, i in enumerate(self.slots):
            if i < 0 or self._loaded_at[slot] == now:
                continue
            self.tiers.put(self._f(i), self._entry(slot, i, st), dirty=True)
            self._loaded_at[slot] = now
        self.tiers.flush()

    def report(self):
        r = self.tiers.report()
        r.update(experts=self._n, resident=self.resident, swaps=self.swaps,
                 vram_params=self.vram_params(), total_params=self.n_params())
        return r
