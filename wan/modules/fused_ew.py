"""Bit-exact Triton fusions of the Wan blocks' elementwise chains (SF_FUSE_EW=1; default off = upstream eager).

RMSNorm  y = (x.float() * rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)).type_as(x) * w
  Eager runs 8 kernels forward (two float32 round trips of the activation) and ~12 in backward; here one kernel each
  way. The two row reductions -- the mean of squares (forward) and the sum of grad * x.float() (the rsqrt branch of
  the backward) -- are added up in exactly the order of ATen's reduce kernel for a float32 reduction over a
  contiguous last dim with >= 16 rows: 32 lanes per row, lane l walks the float4 vectors l, l + 32, ... keeping one
  running sum per vector component, folds them as ((s0 + s1) + s2) + s3, and the 32 lanes are combined by warp
  shuffles at offsets 1, 2, 4, 8, 16 (a pairwise tree); the mean is that sum times float(rows) / float(rows * C)
  (MeanOps). (Fewer than 16 rows makes ATen use 64-lane blocks: those calls stay eager.) Every elementwise step
  rounds where eager rounds (bf16 products through float opmath, the float32 -> bf16 casts) and the kernels are
  built without FP contraction, so outputs and gradients are bit-identical to eager.
  Backward = autograd's formulas for the eager graph: gn = float(bf16(g * w)); gr = sum(gn * a) (a = x.float());
  g_a = gn * r + ((((gr * -0.5) * ((r * r) * r)) * (1 / C)) * (2 * a)); g_x = bf16(g_a). The weight gradient keeps
  the eager reduction: the kernel writes the bf16 product g * y_hat that MulBackward0 forms and sum_to_size
  reduces it like the autograd engine does.

modulate / gate_residual on calls that compiled_ops leaves eager (below SF_COMPILE_MIN_TOKENS, or SF_COMPILE off)
  h * (1 + scale) + shift and x + y * gate in bf16, rounded to bf16 after every op like eager, one kernel instead of
  three / two. In grad mode the backward is eager's formulas (as in compiled_ops).
  SF_EW_LONG=1: every size takes this path (the long sequences too). torch.compile'd functions that are called once
  under a TorchDispatchMode -- SAC's checkpoint context is one -- run eagerly from then on (the frame is skipped for
  good), so with SF_SAC the compiled modulate / gate of SF_COMPILE ran as eager broadcast kernels (~35% of the memory
  bandwidth) in the teacher, the critic and its recompute. Triton launches do not go through the dispatcher, so
  they are unaffected. The backward's two products of g (g * bf16(1 + scale) and g * h; g * gate and g * y) are one
  kernel reading g once; the broadcast-dim sums stay eager on the same bf16 products (same reduction).

Launches: the rollout is CPU-bound, and a plain kernel[grid](...) costs ~23 us of CPU (eager op: ~5.5 us). After
the first launch of a configuration the compiled kernel is called directly (_Launcher), ~5 us.
"""
import atexit
import os
from collections import Counter

import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

ENABLED = os.environ.get('SF_FUSE_EW', '0') == '1'
LONG = ENABLED and os.environ.get('SF_EW_LONG', '0') == '1'   # all sizes + fused backward products
STATS = Counter()   # calls per path, printed at exit: proves the fused path ran (and how often it fell back)
if ENABLED:
    atexit.register(lambda: print(f'[SF_FUSE_EW] {dict(sorted(STATS.items()))}', flush=True))


# ---------------------------------------------------------------------------------------------------------- launch

class _Launcher:
    """kernel[grid](*args, **meta), but later calls with the same constexprs skip Triton's argument binding and
    specialization lookup and call the cached compiled kernel. Only for pointer args that are 16-byte aligned and
    int args declared do_not_specialize, so the cached kernel is the one a fresh lookup would select."""

    def __init__(self, kernel, num_warps):
        self.kernel = kernel
        self.num_warps = num_warps
        self.cache = {}

    def __call__(self, grid, ptrs, scalars, consts):
        stream = torch._C._cuda_getCurrentRawStream(ptrs[0].device.index)
        args = (*ptrs, *scalars, *consts)
        k = self.cache.get((ptrs[0].device.index, consts))
        if k is not None and all(p.data_ptr() % 16 == 0 for p in ptrs):
            k.run(grid[0], grid[1], grid[2], stream, k.function, k.packed_metadata, None, None, None, *args)
            return
        names = self.kernel.arg_names
        meta = dict(zip(names[len(ptrs) + len(scalars):], consts))
        k = self.kernel[grid](*ptrs, *scalars, **meta, num_warps=self.num_warps, enable_fp_fusion=False)
        if all(p.data_ptr() % 16 == 0 for p in ptrs) and k.launch_enter_hook is None:
            self.cache[(ptrs[0].device.index, consts)] = k


# --------------------------------------------------------------------------------------------------------- RMSNorm

@triton.jit
def _lane_tree(s, ROWS: tl.constexpr):
    # s: [ROWS, 32] lane partial sums -> [ROWS]: (2i, 2i+1) pairs first, as warp_shfl_down at offsets 1, 2, 4, 8, 16
    a, b = tl.split(tl.reshape(s, (ROWS, 16, 2)))
    a, b = tl.split(tl.reshape(a + b, (ROWS, 8, 2)))
    a, b = tl.split(tl.reshape(a + b, (ROWS, 4, 2)))
    a, b = tl.split(tl.reshape(a + b, (ROWS, 2, 2)))
    a, b = tl.split(tl.reshape(a + b, (ROWS, 1, 2)))
    return tl.reshape(a + b, (ROWS,))


@triton.jit
def _fold4(acc, ROWS: tl.constexpr):
    # acc: [ROWS, 32, 4] -> ((c0 + c1) + c2) + c3 : [ROWS, 32]
    ev, od = tl.split(tl.reshape(acc, (ROWS, 32, 2, 2)))     # ev: c0, c2   od: c1, c3
    c0, c2 = tl.split(ev)
    c1, c3 = tl.split(od)
    return ((c0 + c1) + c2) + c3


@triton.jit(do_not_specialize=['n_rows'])
def _rms_fwd(x_ptr, w_ptr, y_ptr, r_ptr, n_rows, eps, factor,
             C: tl.constexpr, ROWS: tl.constexpr, SAVE_R: tl.constexpr):
    K: tl.constexpr = C // 128
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    rmask = row < n_rows
    woff = tl.arange(0, 32)[None, :, None] * 4 + tl.arange(0, 4)[None, None, :]
    off = row[:, None, None].to(tl.int64) * C + woff
    m = rmask[:, None, None]
    acc = tl.zeros((ROWS, 32, 4), tl.float32)
    for k in tl.static_range(K):
        a = tl.load(x_ptr + off + k * 128, mask=m, other=0.0).to(tl.float32)
        acc = acc + a * a
    ms = _lane_tree(_fold4(acc, ROWS), ROWS) * factor
    r = libdevice.rsqrt(ms + eps)
    if SAVE_R:
        tl.store(r_ptr + row, r, mask=rmask)
    rb = r[:, None, None]
    for k in tl.static_range(K):
        a = tl.load(x_ptr + off + k * 128, mask=m, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + woff + k * 128).to(tl.float32)
        yh = (a * rb).to(tl.bfloat16)
        tl.store(y_ptr + off + k * 128, (yh.to(tl.float32) * w).to(tl.bfloat16), mask=m)


@triton.jit(do_not_specialize=['n_rows'])
def _rms_bwd(g_ptr, x_ptr, w_ptr, r_ptr, gx_ptr, gwp_ptr, n_rows, inv_c,
             C: tl.constexpr, ROWS: tl.constexpr, NEED_GW: tl.constexpr):
    K: tl.constexpr = C // 128
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    rmask = row < n_rows
    woff = tl.arange(0, 32)[None, :, None] * 4 + tl.arange(0, 4)[None, None, :]
    off = row[:, None, None].to(tl.int64) * C + woff
    m = rmask[:, None, None]
    r = tl.load(r_ptr + row, mask=rmask, other=1.0)
    rb = r[:, None, None]
    acc = tl.zeros((ROWS, 32, 4), tl.float32)
    for k in tl.static_range(K):
        g = tl.load(g_ptr + off + k * 128, mask=m, other=0.0).to(tl.float32)
        a = tl.load(x_ptr + off + k * 128, mask=m, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + woff + k * 128).to(tl.float32)
        gn = (g * w).to(tl.bfloat16).to(tl.float32)            # MulBackward0 (y_hat * w): g * w in bf16, ToCopy -> float
        acc = acc + gn * a                                     # MulBackward0 (a * r), other: (gn * a).sum(-1)
        if NEED_GW:
            yh = (a * rb).to(tl.bfloat16).to(tl.float32)
            tl.store(gwp_ptr + off + k * 128, (g * yh).to(tl.bfloat16), mask=m)
    gr = _lane_tree(_fold4(acc, ROWS), ROWS)
    gs = (gr * -0.5) * ((r * r) * r)                           # RsqrtBackward0: -0.5 * grad * result.pow(3)
    gp = (gs * inv_c)[:, None, None]                           # MeanBackward1: expand / C  (ATen: * (1 / C))
    for k in tl.static_range(K):
        g = tl.load(g_ptr + off + k * 128, mask=m, other=0.0).to(tl.float32)
        a = tl.load(x_ptr + off + k * 128, mask=m, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + woff + k * 128).to(tl.float32)
        gn = (g * w).to(tl.bfloat16).to(tl.float32)
        ga = gn * rb + gp * (a * 2.0)                          # MulBackward0 self-grad + PowBackward0 (grad * 2a)
        tl.store(gx_ptr + off + k * 128, ga.to(tl.bfloat16), mask=m)


_ROWS = 4
_rms_fwd_launch = _Launcher(_rms_fwd, num_warps=4)
_rms_bwd_launch = _Launcher(_rms_bwd, num_warps=4)
_FACTORS = {}


def _f32(v):
    return float(np.float32(v))


def _consts(n, c):
    f = _FACTORS.get((n, c))
    if f is None:
        # MeanOps factor: static_cast<float>(num_output_elements) / numel (float division); MeanBackward: 1 / C
        f = _FACTORS[(n, c)] = (_f32(np.float32(n) / np.float32(n * c)), _f32(np.float32(1.0) / np.float32(c)))
    return f


def _rms_forward(x, weight, eps, r):
    C = x.shape[-1]
    n = x.numel() // C
    y = torch.empty_like(x)
    _rms_fwd_launch((triton.cdiv(n, _ROWS), 1, 1), (x, weight, y, r if r is not None else y),
                    (n, eps, _consts(n, C)[0]), (C, _ROWS, r is not None))
    return y


class _RMSNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, eps):
        r = torch.empty(x.numel() // x.shape[-1], dtype=torch.float32, device=x.device)
        y = _rms_forward(x, weight, eps, r)
        ctx.save_for_backward(x, weight, r)
        return y

    @staticmethod
    def backward(ctx, g):
        x, weight, r = ctx.saved_tensors
        g = g.contiguous()
        C = x.shape[-1]
        n = x.numel() // C
        gx = torch.empty_like(x)
        need_gw = ctx.needs_input_grad[1]
        gwp = torch.empty_like(x) if need_gw else gx
        _rms_bwd_launch((triton.cdiv(n, _ROWS), 1, 1), (g, x, weight, r, gx, gwp), (n, _consts(n, C)[1]),
                        (C, _ROWS, need_gw))
        gw = gwp.sum_to_size(weight.shape) if need_gw else None
        STATS['rms_bwd'] += 1
        return (gx if ctx.needs_input_grad[0] else None), gw, None


def rms_norm(x, weight, eps):
    """WanRMSNorm forward, or None when this case is not covered (the caller runs eager)."""
    if not (x.is_cuda and x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16 and x.is_contiguous()
            and weight.is_contiguous() and weight.dim() == 1 and x.shape[-1] == weight.shape[0]
            and x.shape[-1] % 128 == 0 and x.numel() // x.shape[-1] >= 16):
        STATS['rms_eager'] += 1
        return None
    if torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad):
        STATS['rms_grad'] += 1
        return _RMSNorm.apply(x, weight, eps)
    STATS['rms'] += 1
    return _rms_forward(x, weight, eps, None)


# ------------------------------------------------------------------------------------------ modulate / gate_residual

@triton.jit(do_not_specialize=['S', 'F', 'sb', 'sf', 'tb', 'tf'])
def _mod_kernel(h_ptr, s_ptr, t_ptr, o_ptr, S, F, sb, sf, tb, tf,
                C: tl.constexpr, BS: tl.constexpr, BC: tl.constexpr, GATE: tl.constexpr):
    # MOD : o = bf16(bf16(h * bf16(1 + s)) + t)      h: [B, F, S, C], s/t: [B, F, 1, C] (strided)
    # GATE: o = bf16(h + bf16(s * t))                 h = x, s = y: [B, F, S, C], t = gate: [B, F, 1, C]
    pid_s = tl.program_id(0)
    pid_c = tl.program_id(1)
    bf = tl.program_id(2)
    b = bf // F
    f = bf % F
    srow = pid_s * BS + tl.arange(0, BS)
    c = pid_c * BC + tl.arange(0, BC)
    m = (srow < S)[:, None]
    off = (bf.to(tl.int64) * S + srow[:, None]) * C + c[None, :]
    h = tl.load(h_ptr + off, mask=m, other=0.0).to(tl.float32)
    if GATE:
        y = tl.load(s_ptr + off, mask=m, other=0.0).to(tl.float32)
        gate = tl.load(t_ptr + b * tb + f * tf + c).to(tl.float32)[None, :]
        p = (y * gate).to(tl.bfloat16).to(tl.float32)
        o = (h + p).to(tl.bfloat16)
    else:
        scale = tl.load(s_ptr + b * sb + f * sf + c).to(tl.float32)
        shift = tl.load(t_ptr + b * tb + f * tf + c).to(tl.float32)[None, :]
        one_p = (scale + 1.0).to(tl.bfloat16).to(tl.float32)[None, :]
        mm = (h * one_p).to(tl.bfloat16).to(tl.float32)
        o = (mm + shift).to(tl.bfloat16)
    tl.store(o_ptr + off, o, mask=m)


_mod_launch = _Launcher(_mod_kernel, num_warps=4)
_BS, _BC = 16, 256


@triton.jit(do_not_specialize=['S', 'F', 'vb', 'vf'])
def _mul2_kernel(g_ptr, a_ptr, v_ptr, o1_ptr, o2_ptr, S, F, vb, vf,
                 C: tl.constexpr, BS: tl.constexpr, BC: tl.constexpr, PLUS1: tl.constexpr):
    # o1 = bf16(g * v')  with v' = bf16(1 + v) (PLUS1) or v: the broadcast row vector [B, F, 1, C] (strided)
    # o2 = bf16(g * a)   a: same shape as g.  Products of two bf16 values are exact in float32: no FMA question.
    pid_s = tl.program_id(0)
    pid_c = tl.program_id(1)
    bf = tl.program_id(2)
    b = bf // F
    f = bf % F
    srow = pid_s * BS + tl.arange(0, BS)
    c = pid_c * BC + tl.arange(0, BC)
    m = (srow < S)[:, None]
    off = (bf.to(tl.int64) * S + srow[:, None]) * C + c[None, :]
    g = tl.load(g_ptr + off, mask=m, other=0.0).to(tl.float32)
    a = tl.load(a_ptr + off, mask=m, other=0.0).to(tl.float32)
    v = tl.load(v_ptr + b * vb + f * vf + c).to(tl.float32)
    if PLUS1:
        v = (v + 1.0).to(tl.bfloat16).to(tl.float32)
    tl.store(o1_ptr + off, (g * v[None, :]).to(tl.bfloat16), mask=m)
    tl.store(o2_ptr + off, (g * a).to(tl.bfloat16), mask=m)


_mul2_launch = _Launcher(_mul2_kernel, num_warps=4)


def _mul2(g, a, v, geom, plus1):
    """(bf16(g * v'), bf16(g * a)) in one pass, or None (caller runs eager) when g / a are not plain [B, (F,) S, C]."""
    B, F, S, C, st = geom
    if not LONG or g.shape != a.shape or g.dtype is not _BF16 or not a.is_contiguous():
        return None
    g = g.contiguous()
    o1 = torch.empty_like(g)
    o2 = torch.empty_like(g)
    _mul2_launch((triton.cdiv(S, _BS), C // _BC, B * F), (g, a, v, o1, o2), (S, F, st[0], st[1]), (C, _BS, _BC, plus1))
    STATS['mul2'] += 1
    return o1, o2


_BF16 = torch.bfloat16


def _geom(h, *rows):
    """[B, F, S, C] / [B, S, C] activation with [B, F, 1, C] / [B, 1, C] per-row vectors -> (B, F, S, C, strides)
    or None."""
    if h.dtype is not _BF16 or not h.is_cuda or not h.is_contiguous():
        return None
    nd = h.dim()
    if nd == 4:
        B, F, S, C = h.shape
    elif nd == 3:
        B, S, C = h.shape
        F = 1
    else:
        return None
    if C % _BC:
        return None
    dev = h.get_device()
    strides = []
    for v in rows:
        if v.dtype is not _BF16 or v.dim() != nd or v.get_device() != dev:
            return None
        st = v.stride()
        if nd == 4:
            if v.shape != (B, F, 1, C) or st[3] != 1:
                return None
            strides += [st[0], st[1]]
        else:
            if v.shape != (B, 1, C) or st[2] != 1:
                return None
            strides += [st[0], 0]
    return B, F, S, C, strides


def _mod_forward(h, s, t, geom, gate):
    B, F, S, C, st = geom
    o = torch.empty_like(h)
    _mod_launch((triton.cdiv(S, _BS), C // _BC, B * F), (h, s, t, o),
                (S, F, *(st if not gate else (0, 0, *st)), ), (C, _BS, _BC, gate))
    return o


class _Modulate(torch.autograd.Function):
    """h * (1 + scale) + shift; backward = AddBackward0 -> MulBackward0 -> AddBackward1 as autograd runs them."""

    @staticmethod
    def forward(ctx, h, scale, shift, geom):
        ctx.save_for_backward(h, scale)
        ctx.shift_shape = shift.shape
        ctx.geom = geom
        return _mod_forward(h, scale, shift, geom, False)

    @staticmethod
    def backward(ctx, g):
        h, scale = ctx.saved_tensors
        gh = gs = gsh = None
        if ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
            r = _mul2(g, h, scale, (*ctx.geom[:4], _row_strides(scale, ctx.geom)), True)
            if r is not None:
                gh, gp = r
                gs = gp.sum_to_size(scale.shape)
                if ctx.needs_input_grad[2]:
                    gsh = g.sum_to_size(ctx.shift_shape)
                return gh, gs, gsh, None
        if ctx.needs_input_grad[0]:
            gh = g * (1 + scale)
        if ctx.needs_input_grad[1]:
            gs = (g * h).sum_to_size(scale.shape)
        if ctx.needs_input_grad[2]:
            gsh = g.sum_to_size(ctx.shift_shape)
        return gh, gs, gsh, None


class _GateResidual(torch.autograd.Function):
    """x + y * gate; backward = AddBackward0 -> MulBackward0 as autograd runs them."""

    @staticmethod
    def forward(ctx, x, y, gate, geom):
        ctx.save_for_backward(y, gate)
        ctx.x_shape = x.shape
        ctx.geom = geom
        return _mod_forward(x, y, gate, geom, True)

    @staticmethod
    def backward(ctx, g):
        y, gate = ctx.saved_tensors
        gx = gy = gg = None
        if ctx.needs_input_grad[1] and ctx.needs_input_grad[2]:
            r = _mul2(g, y, gate, (*ctx.geom[:4], _row_strides(gate, ctx.geom)), False)
            if r is not None:
                gy, gq = r
                gg = gq.sum_to_size(gate.shape)
                if ctx.needs_input_grad[0]:
                    gx = g.sum_to_size(ctx.x_shape)
                return gx, gy, gg, None
        if ctx.needs_input_grad[0]:
            gx = g.sum_to_size(ctx.x_shape)
        if ctx.needs_input_grad[1]:
            gy = (g * gate).sum_to_size(y.shape)
        if ctx.needs_input_grad[2]:
            gg = (g * y).sum_to_size(gate.shape)
        return gx, gy, gg, None


def _row_strides(v, geom):
    """(batch, frame) strides of a [B, F, 1, C] / [B, 1, C] row vector, as _geom computes them."""
    st = v.stride()
    return [st[0], st[1]] if v.dim() == 4 else [st[0], 0]


def _grad(*ts):
    return torch.is_grad_enabled() and any(t.requires_grad for t in ts)


def modulate(h, scale, shift):
    """h * (1 + scale) + shift, or None when not covered."""
    geom = _geom(h, scale, shift)
    if geom is None:
        STATS['modulate_eager'] += 1
        return None
    if _grad(h, scale, shift):
        STATS['modulate_grad'] += 1
        return _Modulate.apply(h, scale, shift, geom)
    STATS['modulate'] += 1
    return _mod_forward(h, scale, shift, geom, False)


def gate_residual(x, y, gate):
    """x + y * gate, or None when not covered."""
    geom = _geom(x, gate)
    if geom is None or y.shape != x.shape or y.dtype != torch.bfloat16 or not y.is_contiguous():
        STATS['gate_eager'] += 1
        return None
    if _grad(x, y, gate):
        STATS['gate_grad'] += 1
        return _GateResidual.apply(x, y, gate, geom)
    STATS['gate'] += 1
    return _mod_forward(x, y, gate, geom, True)
