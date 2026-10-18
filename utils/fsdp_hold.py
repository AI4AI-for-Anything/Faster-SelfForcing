"""Rollout-time helpers for the FSDP-wrapped causal generator.

SF_FSDP_HOLD=1 -- one all-gather per rollout instead of one per forward.
The generator is a single FSDP unit (HYBRID_SHARD, use_orig_params): every forward all-gathers the 1.3B parameters and
its post-forward frees them again, ~25 times per rollout, and the parameters do not change during a rollout. Inside
`hold(generator)` the post-forward reshard keeps the padded unsharded flat parameter allocated (FSDP's own no-free path,
the one it takes for a root unit or a ZeRO-2 strategy), so the next forward's unshard finds it in place and skips the
all-gather and the low-precision cast. A forward run with grad enabled (the backpropagated exit step) first frees it
and goes through FSDP's normal all-gather; leaving the rollout frees it as well. The sharding itself (strategy, wrap,
use_orig_params, the per-rank flat-parameter shards) is untouched; only when the unsharded copy is freed changes.

SF_XATTN_CACHE=1 -- make the rollout's cross-attention cache effective.
The FSDP root copies dict and list arguments (torch.distributed.utils._recursive_to), so the pipeline's crossattn_cache
dicts never see is_init=True and every forward recomputes the 30 context K/V projections. CacheSlot is a mapping that
is not a dict, so the copy passes it through by reference: K/V are computed by the first no-grad forward of the
rollout and reused, the same values as recomputing them.
"""
import contextlib
import os
from collections import Counter

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import _runtime_utils
from torch.distributed.fsdp import fully_sharded_data_parallel as _fsdp_module

STATS = Counter()
# EPOCH counts hold() entries (one per rollout) and ACTIVE is the hold() nesting depth: the generator parameters do not
# change inside a rollout, so SF_CUDAGRAPH refreshes its static parameter copy once per EPOCH (utils/cuda_graph.py)
EPOCH = 0
ACTIVE = 0
_orig_post_forward_reshard = _runtime_utils._post_forward_reshard
_installed = False


def hold_enabled():
    return os.environ.get('SF_FSDP_HOLD') == '1'


def xattn_cache_enabled():
    return os.environ.get('SF_XATTN_CACHE') == '1'


def _post_forward_reshard(state, handle):
    if handle is not None and getattr(handle, '_sf_hold', False):
        _runtime_utils._reshard(state, handle, False)     # switch to the shard, keep the unsharded copy allocated
        return
    _orig_post_forward_reshard(state, handle)


def _install():
    global _installed
    if not _installed:
        # FullyShardedDataParallel.forward passes the name it imported; the composable path uses _runtime_utils'
        _runtime_utils._post_forward_reshard = _post_forward_reshard
        _fsdp_module._post_forward_reshard = _post_forward_reshard
        _installed = True


def _release(pairs):
    for state, handle in pairs:
        if handle.uses_sharded_strategy and not handle.needs_unshard():
            _runtime_utils._reshard(state, handle, True)
            STATS['release'] += 1


@contextlib.contextmanager
def hold(module):
    if not hold_enabled() or not isinstance(module, FSDP):
        yield
        return
    _install()
    pairs = [(m, m._handle) for m in FSDP.fsdp_modules(module) if m._handle is not None]

    def before_forward(_module, _args, _kwargs):
        if torch.is_grad_enabled():                     # a backpropagated forward takes FSDP's normal path
            _release(pairs)

    hook = module.register_forward_pre_hook(before_forward, with_kwargs=True)
    for _, handle in pairs:
        handle._sf_hold = True
    STATS['hold'] += 1
    global EPOCH, ACTIVE
    EPOCH += 1
    ACTIVE += 1
    try:
        yield
    finally:
        ACTIVE -= 1
        hook.remove()
        for _, handle in pairs:
            handle._sf_hold = False
        _release(pairs)


def resident_enabled():
    return os.environ.get('SF_T5_RESIDENT') == '1'


def _needs_unshard(handle):
    try:
        return handle.needs_unshard()
    except AttributeError:      # before FSDP's lazy init (first forward): nothing is unsharded yet
        return True


def make_resident(module):
    """SF_T5_RESIDENT=1: a frozen FSDP module that only ever runs without grad (the UMT5 text encoder) keeps its
    unsharded parameters after every forward -- the hold() mechanism without a release -- so only its first forward
    all-gathers. The sharding (strategy, wrap, use_orig_params, the per-rank shards) is unchanged; the unsharded copy
    costs the full bf16 parameter size per GPU (UMT5-XXL encoder: ~11 GB). Counters (printed at exit): forwards and
    FSDP units that still needed an all-gather at forward entry."""
    if not resident_enabled() or not isinstance(module, FSDP):
        return
    _install()
    handles = [m._handle for m in FSDP.fsdp_modules(module) if m._handle is not None]
    for handle in handles:
        handle._sf_hold = True

    def count(_module, _args, _kwargs):
        if torch.is_grad_enabled():
            raise RuntimeError('SF_T5_RESIDENT: the resident module ran with grad enabled')
        STATS['resident_forward'] += 1
        STATS['resident_units_gathered'] += sum(1 for h in handles if _needs_unshard(h))

    module.register_forward_pre_hook(count, with_kwargs=True)
    STATS['resident_units'] += len(handles)
    import atexit
    atexit.register(lambda: print(f'[SF_T5_RESIDENT] {dict((k, v) for k, v in STATS.items() if k.startswith("resident"))}',
                                  flush=True))


class CacheSlot:
    """One block's cross-attention cache: a mapping that FSDP's argument copying passes through by reference."""
    __slots__ = ('entries',)

    def __init__(self, entries):
        self.entries = dict(entries)

    def __getitem__(self, key):
        return self.entries[key]

    def __setitem__(self, key, value):
        self.entries[key] = value

    def get(self, key, default=None):
        return self.entries.get(key, default)
