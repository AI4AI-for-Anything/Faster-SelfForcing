"""Bit-exact fused LayerNorm + modulation for the Wan blocks (SF_FUSE_LNMOD=1; default off = upstream).

norm1 / norm2 (WanLayerNorm, no affine) feed straight into modulate: h = LN(x); o = h * (1 + scale) + shift, with
h making a full round trip through memory. One Triton kernel here does both:

  LayerNorm  ATen's vectorized_layer_norm_kernel (bf16 in, float accumulation) replicated: one row per 128 virtual
             threads (lane + 32 * warp), thread i walks the float4 vectors i, i + 128, ... with the per-element
             Welford update mean = fma(delta, 1/n, mean), m2 = fma(delta, x - mean, m2) (1/n correctly rounded,
             as nvcc builds 1.f/count without fast-math); the 32 lanes combine like warp_shfl_down at offsets
             16, 8, 4, 2, 1 (cuWelfordCombine(self, other)), then the 4 warps at offsets 2, 1; var = m2 / N
             (correctly rounded), rstd = rsqrt(var + eps), h = bf16(rstd * (x - mean)). Every thread holds the
             same count (N a multiple of 512), so the combine weights are equal: n_A = n_B = c * (1 / 2c).
  modulate   o = bf16(bf16(h * bf16(1 + scale)) + shift), rounding after every op like eager (fused_ew._mod_kernel).

Built without FP contraction; the FMAs nvcc formed are written out. Probe (runs/r08/normfuse/ln_probe*.py): h, mean
and rstd bitwise equal to aten.native_layer_norm over row counts 1..32760, N 1536 / 5120, scales 1e-4..3e4, zero and
constant rows. In grad mode the kernel also writes h, mean and rstd and the backward is autograd's for the eager
graph: modulate's formulas (g * (1 + scale), (g * h).sum_to_size, g.sum_to_size), then aten.native_layer_norm_backward
with the saved statistics. Only where ATen itself would take the vectorized kernel (contiguous, 8-byte aligned,
N % 512 == 0); anything else returns None and the caller runs norm + modulate.
"""
import atexit
import os
from collections import Counter

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from .fused_ew import _Launcher

ENABLED = os.environ.get('SF_FUSE_LNMOD', '0') == '1'
STATS = Counter()
if ENABLED:
    atexit.register(lambda: print(f'[SF_FUSE_LNMOD] {dict(sorted(STATS.items()))}', flush=True))


@triton.jit
def _comb(m_s, s_s, m_o, s_o, c: tl.constexpr):
    # cuWelfordCombine(dataB = self, dataA = other) with equal counts c: count 2c, coef = 1/(2c), n_A = n_B = c * coef,
    # which is exactly 0.5 in float32 for every count used here (checked on the host, _check_constants)
    delta = m_s - m_o
    mean = tl.fma(m_o, 0.5, m_s * 0.5)
    sig = tl.fma((delta * delta) * c, 0.5, s_o + s_s)
    return mean, sig


@triton.jit
def _pairs(m, s, c: tl.constexpr, R: tl.constexpr, H: tl.constexpr):
    # [R, 2H] -> [R, H]: position 2k (self) combines with position 2k + 1 (other)
    m_s, m_o = tl.split(tl.reshape(m, (R, H, 2)))
    s_s, s_o = tl.split(tl.reshape(s, (R, H, 2)))
    return _comb(m_s, s_s, m_o, s_o, c)


@triton.jit
def _bitrev(v, BITS: tl.constexpr):
    r = v * 0
    for i in tl.static_range(BITS):
        r = r | (((v >> i) & 1) << (BITS - 1 - i))
    return r


@triton.jit(do_not_specialize=['n_rows', 'L', 'S', 'sb', 'sf', 'tb', 'tf'])
def _ln_mod_fwd(x_ptr, s_ptr, t_ptr, o_ptr, h_ptr, mean_ptr, rstd_ptr, n_rows, L, S, sb, sf, tb, tf, eps, Nf,
                N: tl.constexpr, ROWS: tl.constexpr, SAVE: tl.constexpr):
    NV: tl.constexpr = N // 512
    C0: tl.constexpr = NV * 4.0
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    rmask = row < n_rows
    # position p holds ATen thread thrx = lane + 32 * warp with lane = bitrev5(p % 32), warp = bitrev2(p // 32):
    # then every step of ATen's combine tree (lanes at shuffle-down offsets 16 .. 1, then warps at 2, 1) pairs two
    # adjacent positions
    p = tl.arange(0, 128)
    thr = _bitrev(p % 32, 5) + 32 * _bitrev(p // 32, 2)
    ii = tl.arange(0, 4)
    base = row[:, None, None].to(tl.int64) * N
    m3 = rmask[:, None, None]
    mean = tl.zeros((ROWS, 128), tl.float32)
    sig = tl.zeros((ROWS, 128), tl.float32)
    for v in tl.static_range(NV):
        xv = tl.load(x_ptr + base + (thr[None, :, None] + 128 * v) * 4 + ii[None, None, :], mask=m3,
                     other=0.0).to(tl.float32)                                       # [ROWS, 128, 4]
        a, b = tl.split(tl.reshape(xv, (ROWS, 128, 2, 2)))                          # a: cols 0, 2  b: cols 1, 3
        x0, x2 = tl.split(a)
        x1, x3 = tl.split(b)
        for q in tl.static_range(4):
            if q == 0:
                val = x0
            elif q == 1:
                val = x1
            elif q == 2:
                val = x2
            else:
                val = x3
            delta = val - mean
            # 1.f / new_count: a compile-time constant equal to the float32 quotient (checked, _check_constants)
            mean = tl.fma(delta, 1.0 / (v * 4 + q + 1), mean)
            sig = tl.fma(delta, val - mean, sig)
    mean, sig = _pairs(mean, sig, C0, ROWS, 64)
    mean, sig = _pairs(mean, sig, C0 * 2, ROWS, 32)
    mean, sig = _pairs(mean, sig, C0 * 4, ROWS, 16)
    mean, sig = _pairs(mean, sig, C0 * 8, ROWS, 8)
    mean, sig = _pairs(mean, sig, C0 * 16, ROWS, 4)
    mean, sig = _pairs(mean, sig, C0 * 32, ROWS, 2)
    mean, sig = _pairs(mean, sig, C0 * 64, ROWS, 1)
    mrow = tl.reshape(mean, (ROWS,))
    var = libdevice.div_rn(tl.reshape(sig, (ROWS,)), Nf)
    rstd = libdevice.rsqrt(var + eps)
    if SAVE:
        tl.store(mean_ptr + row, mrow, mask=rmask)
        tl.store(rstd_ptr + row, rstd, mask=rmask)
    # modulation rows: token t of sample b is in frame t // S; scale / shift are [B, F, 1, N] (strides sb, sf)
    bi = row // L
    fi = (row % L) // S
    soff = (bi * sb + fi * sf)[:, None]
    toff = (bi * tb + fi * tf)[:, None]
    mb = mrow[:, None]
    rb = rstd[:, None]
    base2 = row[:, None].to(tl.int64) * N
    m2 = rmask[:, None]
    for v in tl.static_range(NV):
        col = tl.arange(0, 512)[None, :] + 512 * v
        xv = tl.load(x_ptr + base2 + col, mask=m2, other=0.0).to(tl.float32)
        h = (rb * (xv - mb)).to(tl.bfloat16)
        if SAVE:
            tl.store(h_ptr + base2 + col, h, mask=m2)
        sc = tl.load(s_ptr + soff + col, mask=m2, other=0.0).to(tl.float32)
        sh = tl.load(t_ptr + toff + col, mask=m2, other=0.0).to(tl.float32)
        one_p = (sc + 1.0).to(tl.bfloat16).to(tl.float32)
        mm = (h.to(tl.float32) * one_p).to(tl.bfloat16).to(tl.float32)
        tl.store(o_ptr + base2 + col, (mm + sh).to(tl.bfloat16), mask=m2)


_ROWS = 1   # one row per single-warp program: measured fastest (rows 4680 / 32760, N 1536 / 5120)


def _check_constants(n):
    """The kernel's compile-time constants must be the float32 values nvcc's code computes at run time."""
    import numpy as np
    for k in range(1, n // 128 + 1):
        assert np.float32(1.0 / k) == np.float32(1) / np.float32(k), k
    c = np.float32(n // 128)
    for _ in range(7):
        assert c * (np.float32(1) / (c + c)) == np.float32(0.5), c
        c = c + c
    return True


_CHECKED = {}
_fwd_launch = _Launcher(_ln_mod_fwd, num_warps=1)


def _launch(x, s, t, geom, eps, save):
    B, L, S, N, sb, sf, tb, tf = geom
    n = B * L
    o = torch.empty_like(x)
    if save:
        h = torch.empty_like(x)
        mean = torch.empty((B, L, 1), dtype=torch.float32, device=x.device)
        rstd = torch.empty((B, L, 1), dtype=torch.float32, device=x.device)
    else:
        h = mean = rstd = o
    _fwd_launch((triton.cdiv(n, _ROWS), 1, 1), (x, s, t, o, h, mean, rstd), (n, L, S, sb, sf, tb, tf, eps, float(N)),
                (N, _ROWS, save))
    return o, h, mean, rstd


class _LnMod(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, scale, shift, eps, geom, out_shape):
        o, h, mean, rstd = _launch(x, scale, shift, geom, eps, True)
        ctx.save_for_backward(x, mean, rstd, h.view(out_shape), scale)
        ctx.shift_shape = shift.shape
        return o.view(out_shape)

    @staticmethod
    def backward(ctx, g):
        x, mean, rstd, h, scale = ctx.saved_tensors
        gx = gs = gsh = None
        if ctx.needs_input_grad[1]:
            gs = (g * h).sum_to_size(scale.shape)              # modulate: MulBackward0 (other) -> AddBackward
        if ctx.needs_input_grad[2]:
            gsh = g.sum_to_size(ctx.shift_shape)               # AddBackward0 (shift)
        if ctx.needs_input_grad[0]:
            gh = (g * (1 + scale)).reshape(x.shape)            # MulBackward0 (self), then the unflatten view
            gx = torch.ops.aten.native_layer_norm_backward(gh, x, [x.shape[-1]], mean, rstd, None, None,
                                                           [True, False, False])[0]
        STATS['bwd'] += 1
        return gx, gs, gsh, None, None, None


def ln_modulate(norm, x, scale, shift, frames=None):
    """modulate(norm(x)[.unflatten(1, frames)], scale, shift) for a WanLayerNorm without affine, or None."""
    if not (norm.weight is None and norm.bias is None and x.is_cuda and x.dtype == torch.bfloat16 and x.dim() == 3
            and x.is_contiguous() and x.data_ptr() % 16 == 0 and x.shape[-1] % 512 == 0
            and tuple(norm.normalized_shape) == (x.shape[-1],)):
        STATS['uncovered'] += 1
        return None
    B, L, N = x.shape
    if N not in _CHECKED:
        _CHECKED[N] = _check_constants(N)
    if frames is None:
        F, S = 1, L
        out_shape = x.shape
        want = (B, 1, N)
    else:
        F, S = frames
        out_shape = (B, F, S, N)
        want = (B, F, 1, N)
    if F * S != L:
        STATS['uncovered'] += 1
        return None
    st = []
    for v in (scale, shift):
        if not (v.is_cuda and v.dtype == torch.bfloat16 and tuple(v.shape) == want and v.stride(-1) == 1
                and v.data_ptr() % 16 == 0):
            STATS['uncovered'] += 1
            return None
        st += [v.stride(0), v.stride(1) if frames is not None else 0]
    geom = (B, L, S, N, *st)
    if torch.is_grad_enabled() and (x.requires_grad or scale.requires_grad or shift.requires_grad):
        STATS['grad'] += 1
        return _LnMod.apply(x, scale, shift, norm.eps, geom, out_shape)
    STATS['nograd'] += 1
    return _launch(x, scale, shift, geom, norm.eps, False)[0].view(out_shape)
