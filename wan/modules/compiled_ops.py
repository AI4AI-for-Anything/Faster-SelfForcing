"""Selective torch.compile of the pointwise chains in the Wan blocks (SF_COMPILE=1; default off = upstream eager).

Only pointwise chains are compiled, with inductor's emulate_precision_casts, so every intermediate that eager rounds
to bf16 is rounded at the same place: the compiled forward is meant to be bitwise equal to eager. Reductions (the
layer norms, the RMSNorm mean square, gradient sums over broadcast dims) stay eager, so their reduction order is the
upstream one. In grad mode the autograd functions below run the compiled forward and, in backward, the same eager
formulas autograd runs for the original expression. RMSNorm is compiled only without grad.
Shapes are static (dynamic=False): rollout blocks (3 frames x 1560 tokens) and full sequences (21 x 1560), dims
1536 / 5120. Inputs are detached before the compiled call so grad and no-grad calls share one graph per shape.
"""
import os

import torch

from . import fused_ew

ENABLED = os.environ.get('SF_COMPILE', '0') == '1'
# calls on fewer tokens than this stay eager: a compiled call costs ~35 us of CPU (~3 eager ops), which loses on the
# launch-bound rollout (3 frames = 4680 tokens) and wins on the GPU-bound full sequences (32760 tokens)
MIN_TOKENS = int(os.environ.get('SF_COMPILE_MIN_TOKENS', '0'))
_compiled = {}


def _modulate_impl(h, scale, shift):
    return h * (1 + scale) + shift


def _gate_residual_impl(x, y, gate):
    return x + y * gate


def _rms_square_impl(x):
    xf = x.float()
    return xf * xf  # == x.float().pow(2): ATen's pow kernel computes base * base for exponent 2


def _rms_scale_impl(x, ms, weight, eps):
    # rsqrt(ms + eps) inside the kernel is bitwise equal to ATen's add + rsqrt kernels (checked on 5 value ranges)
    return (x.float() * torch.rsqrt(ms + eps)).type_as(x) * weight


def _c(fn):
    """torch.compile(fn) with eager bf16 rounding points (emulate_precision_casts) and without FP contraction.

    Triton contracts a*b + c into an FMA even across the bf16 round trip that emulate_precision_casts inserts, so
    the kernels are built with TRITON_DEFAULT_FP_FUSION=0 (also passes --fmad=false to ptxas). The variable is set
    only around these calls and compilation runs in this process (compile_threads=1), so other Triton kernels in the
    process keep their own setting; the option is part of Triton's cache key.
    """
    f = _compiled.get(fn)
    if f is None:
        import torch._dynamo
        torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 16)
        cf = torch.compile(fn, dynamic=False, fullgraph=True,
                           options={'emulate_precision_casts': True, 'compile_threads': 1})

        def f(*args):
            if fused_ew.LONG and torch._C._len_torch_dispatch_stack():
                # SF_EW_LONG: under a TorchDispatchMode (SAC) dynamo would skip this frame for good -- run eager
                return fn(*args)
            old = os.environ.get('TRITON_DEFAULT_FP_FUSION')
            os.environ['TRITON_DEFAULT_FP_FUSION'] = '0'
            try:
                return cf(*args)
            finally:
                if old is None:
                    del os.environ['TRITON_DEFAULT_FP_FUSION']
                else:
                    os.environ['TRITON_DEFAULT_FP_FUSION'] = old
        _compiled[fn] = f
    return f


class _Modulate(torch.autograd.Function):
    """h * (1 + scale) + shift; backward = AddBackward0 -> MulBackward0 -> AddBackward1 as autograd runs them."""

    @staticmethod
    def forward(ctx, h, scale, shift):
        ctx.save_for_backward(h, scale)
        ctx.shift_shape = shift.shape
        return _c(_modulate_impl)(h.detach(), scale.detach(), shift.detach())

    @staticmethod
    def backward(ctx, g):
        h, scale = ctx.saved_tensors
        gh = gs = gsh = None
        if ctx.needs_input_grad[0]:
            gh = g * (1 + scale)
        if ctx.needs_input_grad[1]:
            gs = (g * h).sum_to_size(scale.shape)
        if ctx.needs_input_grad[2]:
            gsh = g.sum_to_size(ctx.shift_shape)
        return gh, gs, gsh


class _GateResidual(torch.autograd.Function):
    """x + y * gate; backward = AddBackward0 -> MulBackward0 as autograd runs them."""

    @staticmethod
    def forward(ctx, x, y, gate):
        ctx.save_for_backward(y, gate)
        ctx.x_shape = x.shape
        return _c(_gate_residual_impl)(x.detach(), y.detach(), gate.detach())

    @staticmethod
    def backward(ctx, g):
        y, gate = ctx.saved_tensors
        gx = gy = gg = None
        if ctx.needs_input_grad[0]:
            gx = g.sum_to_size(ctx.x_shape)
        if ctx.needs_input_grad[1]:
            gy = (g * gate).sum_to_size(y.shape)
        if ctx.needs_input_grad[2]:
            gg = (g * y).sum_to_size(gate.shape)
        return gx, gy, gg


def _small(x):
    return MIN_TOKENS > 0 and x.numel() // x.shape[-1] < MIN_TOKENS


def _needs_grad(*ts):
    return torch.is_grad_enabled() and any(t.requires_grad for t in ts)


def modulate(h, scale, shift):
    if fused_ew.LONG:  # SF_EW_LONG=1: every size through the Triton kernel (see fused_ew)
        out = fused_ew.modulate(h, scale, shift)
        if out is not None:
            return out
    if not ENABLED or _small(h):
        if fused_ew.ENABLED:  # SF_FUSE_EW=1: the calls left eager here run as one bit-identical Triton kernel
            out = fused_ew.modulate(h, scale, shift)
            if out is not None:
                return out
        return h * (1 + scale) + shift
    if _needs_grad(h, scale, shift):
        return _Modulate.apply(h, scale, shift)
    return _c(_modulate_impl)(h, scale, shift)


def gate_residual(x, y, gate):
    if fused_ew.LONG:
        out = fused_ew.gate_residual(x, y, gate)
        if out is not None:
            return out
    if not ENABLED or _small(x):
        if fused_ew.ENABLED:
            out = fused_ew.gate_residual(x, y, gate)
            if out is not None:
                return out
        return x + y * gate
    if _needs_grad(x, y, gate):
        return _GateResidual.apply(x, y, gate)
    return _c(_gate_residual_impl)(x, y, gate)


def rms_norm(x, weight, eps):
    """WanRMSNorm without grad in 3 kernels (eager: 8): x.float()**2, the eager mean (same reduction order),
    then rsqrt(ms + eps) * x -> type_as -> * weight."""
    sq = _c(_rms_square_impl)(x)
    return _c(_rms_scale_impl)(x, sq.mean(dim=-1, keepdim=True), weight, eps)


def rms_norm_usable(x, weight):
    return ENABLED and not _small(x) and not _needs_grad(x, weight)
