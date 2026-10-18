"""Selective activation checkpointing for the Wan blocks.

SF_SAC=1 (all checkpointed Wan blocks) or SF_SAC=bidir (bidirectional WanModel only); unset/0 = plain checkpointing.

The blocks run under torch.utils.checkpoint (non-reentrant), so backward recomputes each block's forward. With a
selective-checkpoint context the flash-attention forward ops (FA2 and the vendored FA3, both torch custom ops) keep
their outputs from the first forward (MUST_SAVE) and the recompute takes them from there instead of running
attention again; every other op is recomputed exactly as before. The recomputed attention would be bit-identical
anyway (same inputs, deterministic forward kernel), so the values that reach backward do not change.

SF_SAC_SAVE=<roles> (with SF_SAC on; default empty = only attention): also keep the outputs of these linear layers of
each checkpointed block, so the recompute skips their GEMMs. Roles: q, k, v, o (self-attention), cq, co (cross-attention
query / output), ffn1, ffn2. SF_SAC_SAVE_SCOPE=bidir (default) | causal | all picks the models it applies to. A role is
read off the GEMM itself: ffn1 / ffn2 by shape (dim -> ffn_dim / ffn_dim -> dim); the dim -> dim GEMMs on the block's
tokens by their order (q, k, v, o, cq, co), skipping the cross-attention K/V GEMMs on the text context (other row
count). The forward and its recompute dispatch the same ops in the same order, so both passes assign the same roles.
Each saved role costs tokens x out_features bf16 per block (dim -> dim: 100 MB per block at 32760 tokens).
"""
import atexit
import os
from collections import Counter
from functools import partial

import torch
from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts

MODE = os.environ.get('SF_SAC', '0')
SAVE = frozenset(r for r in os.environ.get('SF_SAC_SAVE', '').split(',') if r)
SAVE_SCOPE = os.environ.get('SF_SAC_SAVE_SCOPE', 'bidir')
_ROLES = ('q', 'k', 'v', 'o', 'cq', 'co')     # the block's dim -> dim GEMMs on its own tokens, in call order
_KNOWN = frozenset(_ROLES + ('ffn1', 'ffn2'))
assert SAVE <= _KNOWN, f'SF_SAC_SAVE: unknown roles {sorted(SAVE - _KNOWN)}'
_ATTN_FWD = (('flash_attn', '_flash_attn_forward'), ('flash_attn', '_flash_attn_varlen_forward'),
             ('flash_attn_3', '_flash_attn_forward'))
_GEMMS = (torch.ops.aten.addmm.default, torch.ops.aten.mm.default)
_saved_ops = None
STATS = Counter()   # SF_SAC_SAVE: '<bidir|causal>_<role>_saved' (forward) / '_reused' (recompute), printed at exit
if SAVE:
    atexit.register(lambda: print(f'[SF_SAC_SAVE] {dict(sorted(STATS.items()))}', flush=True))


def _attention_forward_ops():
    ops = set()
    for ns, name in _ATTN_FWD:
        try:
            ops.add(getattr(getattr(torch.ops, ns), name).default)
        except (AttributeError, RuntimeError):
            pass  # that backend is not loaded in this process
    return frozenset(ops)


def _policy(ctx, func, *args, **kwargs):
    return CheckpointPolicy.MUST_SAVE if func in _saved_ops else CheckpointPolicy.PREFER_RECOMPUTE


class _BlockPolicy:
    """_policy plus SF_SAC_SAVE roles, for one checkpointed block call (fresh per call: it counts GEMMs)."""

    def __init__(self, save, kind):
        self.save = save
        self.kind = kind
        self.passes = {False: [None, 0], True: [None, 0]}   # per pass: the block's token rows, dim -> dim GEMMs seen

    def role(self, recompute, func, args):
        a, b = (args[1], args[2]) if func is _GEMMS[0] else (args[0], args[1])
        rows, k, n = a.shape[0], a.shape[1], b.shape[1]
        state = self.passes[recompute]
        if state[0] is None:
            state[0] = rows                                  # the first GEMM of a block is q, on the block's tokens
        if rows != state[0]:
            return None                                      # cross-attention K/V on the text context
        if n > k:
            return 'ffn1'
        if k > n:
            return 'ffn2'
        i = state[1]
        state[1] += 1
        return _ROLES[i] if i < len(_ROLES) else None

    def __call__(self, ctx, func, *args, **kwargs):
        if func in _saved_ops:
            return CheckpointPolicy.MUST_SAVE
        if func in _GEMMS:
            r = self.role(ctx.is_recompute, func, args)
            if r in self.save:
                STATS[f'{self.kind}_{r}_{"reused" if ctx.is_recompute else "saved"}'] += 1
                return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE


def _block_contexts(kind):
    return create_selective_checkpoint_contexts(_BlockPolicy(SAVE, kind))


def checkpoint_kwargs(causal):
    """Extra torch.utils.checkpoint kwargs for one model forward: {} unless SF_SAC selects this model."""
    global _saved_ops
    if MODE == '1' or (MODE == 'bidir' and not causal):
        if _saved_ops is None:
            _saved_ops = _attention_forward_ops()
        if SAVE and SAVE_SCOPE in ('all', 'causal' if causal else 'bidir'):
            return {'context_fn': partial(_block_contexts, 'causal' if causal else 'bidir')}
        return {'context_fn': partial(create_selective_checkpoint_contexts, _policy)}
    return {}
