"""Fused RoPE for rope_apply / causal_rope_apply (SF_ROPE_FUSED=1).

The upstream functions rotate q/k by casting bf16 -> float64, multiplying by the complex128 frequency table
(three column groups indexed by frame / row / column of each token), and casting back. That is ~10 kernels and
several full-size float64 round trips per call. Here one Triton kernel per direction does the same arithmetic in
registers:

  forward   y  = bf16(f32(x * freq))            x read as float64 (exact), complex product in float64
  backward  gx = bf16(f32(gy * conj(freq))) + 0  the "+ 0" is autograd adding the (zero) gradient of the empty
                                                tail slice x[i, seq_len:], which turns -0.0 into +0.0

The complex product uses the same fused-multiply-add pattern as PyTorch's CUDA complex128 multiply (picked by
measuring, see FMA_VARIANT), and float64 -> bfloat16 goes through float32 like c10::BFloat16 does, so results are
bit-identical to the upstream path. Tokens beyond f*h*w (padding tail) pass through unchanged. Anything outside
the covered case (non-CUDA, other dtypes, per-sample grids that differ) returns None and the caller falls back.
"""
import os

import torch
import triton
import triton.language as tl

ROPE_FUSED = os.environ.get('SF_ROPE_FUSED', '0') == '1'
FMA_VARIANT = 1   # see _cmul
_CFG = (32, 4)    # tokens per program, heads per program


@triton.jit
def _cmul(a, b, c, d, VARIANT: tl.constexpr):
    # (a + bi)(c + di) = (ac - bd) + (ad + bc)i with the rounding pattern selected by VARIANT. Negation is a multiply
    # by -1 because Triton lowers unary minus to 0 - x, which turns +0 into +0 instead of -0.
    if VARIANT == 0:
        r = a * c - b * d
        i = a * d + b * c
    elif VARIANT == 1:
        r = tl.fma(a, c, (b * d) * -1.0)
        i = tl.fma(a, d, b * c)
    elif VARIANT == 2:
        r = tl.fma(b * -1.0, d, a * c)
        i = tl.fma(b, c, a * d)
    elif VARIANT == 3:
        r = tl.fma(a, c, (b * d) * -1.0)
        i = tl.fma(b, c, a * d)
    elif VARIANT == 4:
        r = tl.fma(b * -1.0, d, a * c)
        i = tl.fma(a, d, b * c)
    else:  # timing probe only: float32 arithmetic, not bit-exact
        a32 = a.to(tl.float32)
        b32 = b.to(tl.float32)
        c32 = c.to(tl.float32)
        d32 = d.to(tl.float32)
        r = tl.fma(a32, c32, (b32 * d32) * -1.0).to(tl.float64)
        i = tl.fma(a32, d32, b32 * c32).to(tl.float64)
    return r, i


@triton.jit(do_not_specialize=['L', 'seq_len', 'n_heads', 'hw', 'w', 'start_frame'])
def _rope_kernel(x_ptr, y_ptr, tab_ptr, L, seq_len, n_heads, hw, w, start_frame,
                 sxb, sxl, sxh, syb, syl, syh,
                 C: tl.constexpr, C0: tl.constexpr, C1: tl.constexpr, BLOCK_T: tl.constexpr, HB: tl.constexpr,
                 CONJ: tl.constexpr, ADD_ZERO: tl.constexpr, OUT_F64: tl.constexpr, VARIANT: tl.constexpr):
    pid = tl.program_id(0)
    hb = tl.program_id(1)
    b = tl.program_id(2)
    t = pid * BLOCK_T + tl.arange(0, BLOCK_T)[:, None]          # token
    j = tl.arange(0, C)[None, :]                                 # complex pair within the head
    inb = t < L
    rot = inb & (t < seq_len)
    # frequency row of each (token, pair): frame index for the first C0 pairs, row for the next C1, column after
    fi = t // hw + start_frame
    hi = (t % hw) // w
    wi = t % w
    row = tl.where(j < C0, fi, tl.where(j < C0 + C1, hi, wi))
    fo = row * (2 * C) + 2 * j
    c = tl.load(tab_ptr + fo, mask=rot, other=1.0)
    d = tl.load(tab_ptr + fo + 1, mask=rot, other=0.0)
    if CONJ:
        d = d * -1.0
    e = tl.arange(0, 2 * C)[None, :]                             # element within the head (re/im interleaved)
    xo = b * sxb + t * sxl + e
    yo = b * syb + t * syl + e
    for k in tl.static_range(HB):
        hh = hb * HB + k
        m = inb & (hh < n_heads)
        xv = tl.load(x_ptr + xo + hh * sxh, mask=m, other=0.0)
        xr, xi = tl.split(tl.reshape(xv, (BLOCK_T, C, 2)))
        r, i = _cmul(xr.to(tl.float64), xi.to(tl.float64), c, d, VARIANT)
        if OUT_F64:
            orr = r
            oi = i
        else:
            orr = r.to(tl.float32).to(tl.bfloat16)
            oi = i.to(tl.float32).to(tl.bfloat16)
            # padding tail: the upstream cat/stack/type_as round trip is exact, so it is a copy
            orr = tl.where(rot, orr, xr)
            oi = tl.where(rot, oi, xi)
            if ADD_ZERO:
                orr = (orr.to(tl.float32) + 0.0).to(tl.bfloat16)
                oi = (oi.to(tl.float32) + 0.0).to(tl.bfloat16)
        tl.store(y_ptr + yo + hh * syh, tl.reshape(tl.join(orr, oi), (BLOCK_T, 2 * C)), mask=m)


def _launch(x, tab, f, h, w, seq_len, start_frame, conj, add_zero, out=None, variant=None):
    B, L, n, D = x.shape
    C = D // 2
    if out is None:
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    BLOCK_T, HB = _CFG
    grid = (triton.cdiv(L, BLOCK_T), triton.cdiv(n, HB), B)
    _rope_kernel[grid](
        x, out, tab, L, seq_len, n, h * w, w, start_frame,
        x.stride(0), x.stride(1), x.stride(2), out.stride(0), out.stride(1), out.stride(2),
        C=C, C0=C - 2 * (C // 3), C1=C // 3, BLOCK_T=BLOCK_T, HB=HB,
        CONJ=conj, ADD_ZERO=add_zero, OUT_F64=out.dtype == torch.float64,
        VARIANT=FMA_VARIANT if variant is None else variant,
        num_warps=4, enable_fp_fusion=False)
    return out


class _FusedRope(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, tab, f, h, w, seq_len, start_frame):
        ctx.save_for_backward(tab)
        ctx.geom = (f, h, w, seq_len, start_frame)
        return _launch(x, tab, f, h, w, seq_len, start_frame, conj=False, add_zero=False)

    @staticmethod
    def backward(ctx, gy):
        tab, = ctx.saved_tensors
        if gy.stride(-1) != 1:
            gy = gy.contiguous()
        f, h, w, seq_len, start_frame = ctx.geom
        return _launch(gy, tab, f, h, w, seq_len, start_frame, conj=True, add_zero=True), None, None, None, None, \
            None, None


_TABLES = {}


def _table(freqs):
    """float64 view [rows, C, 2] of the complex frequency table (no copy for the usual contiguous complex128)."""
    key = (freqs.data_ptr(), freqs.dtype, tuple(freqs.shape), freqs.device)
    tab = _TABLES.get(key)
    if tab is None:
        tab = torch.view_as_real(freqs.to(torch.complex128).contiguous())
        _TABLES.clear()
        _TABLES[key] = tab
    return tab


def fused_rope_apply(x, grid_sizes, freqs, start_frame=0):
    """Same result as rope_apply / causal_rope_apply, or None when this case is not covered."""
    if not (x.is_cuda and x.dim() == 4 and x.dtype == torch.bfloat16 and x.size(3) % 2 == 0 and x.stride(-1) == 1
            and x.size(3) & (x.size(3) - 1) == 0
            and freqs.is_cuda and freqs.dim() == 2 and freqs.size(1) == x.size(3) // 2 and not grid_sizes.is_cuda):
        return None
    sizes = grid_sizes.tolist()
    if len(sizes) != x.size(0) or any(s != sizes[0] for s in sizes):
        return None
    f, h, w = sizes[0]
    seq_len = f * h * w
    if seq_len > x.size(1) or start_frame + f > freqs.size(0) or h > freqs.size(0) or w > freqs.size(0):
        return None
    return _FusedRope.apply(x, _table(freqs), f, h, w, seq_len, start_frame)
