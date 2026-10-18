"""Bit-exact fused q/k RMSNorm + rotary embedding for the Wan self-attention (SF_FUSE_QKROPE=1; default off).

The self-attention computes q = norm_q(W_q x) (WanRMSNorm) and then rope_apply / causal_rope_apply(q), likewise k.
With SF_FUSE_EW and SF_ROPE_FUSED that is two Triton kernels with the normalised q/k making a full round trip
through memory in between. Here one kernel per direction does both, on the same per-element arithmetic:

  forward   the RMSNorm of fused_ew._rms_fwd (mean of squares in ATen's reduce order, bf16 after x * r and after
            * w), then, still in registers, the rotation of fused_rope._rope_kernel (float64 complex product with
            the same FMA pattern, float64 -> float32 -> bf16, padding tail passed through)
  backward  unchanged: fused_rope's backward kernel, then fused_ew's RMSNorm backward kernel (a single kernel that
            recomputes the rotation in both of the RMSNorm backward passes measured no faster: float64-bound)

Head dim 128 = one 128-element chunk of the RMSNorm row layout (32 lanes x float4), so lane l of chunk h holds the
complex pairs 2l and 2l + 1 of head h. Kernels are built without FP contraction like the two they replace, so
outputs and gradients are bit-identical to them and to eager. Cases outside the covered one (other dtypes or head
dims, < 16 rows, per-sample grids that differ, no qk_norm) return None and the caller runs the unfused path.
"""
import atexit
import os
from collections import Counter

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from .fused_ew import _ROWS as _RMS_ROWS, _Launcher, _consts, _fold4, _lane_tree, _rms_bwd_launch
from .fused_rope import FMA_VARIANT, _cmul, _table
from .fused_rope import _launch as _rope_launch

ENABLED = os.environ.get('SF_FUSE_QKROPE', '0') == '1'
STATS = Counter()
if ENABLED:
    atexit.register(lambda: print(f'[SF_FUSE_QKROPE] {dict(sorted(STATS.items()))}', flush=True))


@triton.jit
def _rot(vr, vi, t, j, rmask, tab_ptr, seq_len, hw, gw, start_frame,
         CP: tl.constexpr, C0: tl.constexpr, C1: tl.constexpr, CONJ: tl.constexpr, ADD_ZERO: tl.constexpr,
         VARIANT: tl.constexpr):
    # vr, vi: [ROWS, 32, 2] bf16 (re / im of pair j); t: [ROWS, 1, 1] token; j: [1, 32, 2] pair within the head.
    # Same arithmetic as fused_rope._rope_kernel.
    rot = rmask & (t < seq_len)
    fi = t // hw + start_frame
    hi = (t % hw) // gw
    wi = t % gw
    row = tl.where(j < C0, fi, tl.where(j < C0 + C1, hi, wi))
    fo = row * (2 * CP) + 2 * j
    c = tl.load(tab_ptr + fo, mask=rot, other=1.0)
    d = tl.load(tab_ptr + fo + 1, mask=rot, other=0.0)
    if CONJ:
        d = d * -1.0
    r, i = _cmul(vr.to(tl.float64), vi.to(tl.float64), c, d, VARIANT)
    orr = r.to(tl.float32).to(tl.bfloat16)
    oi = i.to(tl.float32).to(tl.bfloat16)
    orr = tl.where(rot, orr, vr)
    oi = tl.where(rot, oi, vi)
    if ADD_ZERO:
        orr = (orr.to(tl.float32) + 0.0).to(tl.bfloat16)
        oi = (oi.to(tl.float32) + 0.0).to(tl.bfloat16)
    return orr, oi


@triton.jit(do_not_specialize=['n_rows', 'L', 'seq_len', 'hw', 'gw', 'start_frame'])
def _rms_rope_fwd(x_ptr, w_ptr, tab_ptr, y_ptr, r_ptr, n_rows, L, seq_len, hw, gw, start_frame, eps, factor,
                  C: tl.constexpr, CP: tl.constexpr, C0: tl.constexpr, C1: tl.constexpr, ROWS: tl.constexpr,
                  SAVE_R: tl.constexpr, VARIANT: tl.constexpr):
    K: tl.constexpr = C // 128
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    rmask = row < n_rows
    woff = tl.arange(0, 32)[None, :, None] * 4 + tl.arange(0, 4)[None, None, :]
    off = row[:, None, None].to(tl.int64) * C + woff
    m = rmask[:, None, None]
    # RMSNorm statistics: fused_ew._rms_fwd
    acc = tl.zeros((ROWS, 32, 4), tl.float32)
    for k in tl.static_range(K):
        a = tl.load(x_ptr + off + k * 128, mask=m, other=0.0).to(tl.float32)
        acc = acc + a * a
    ms = _lane_tree(_fold4(acc, ROWS), ROWS) * factor
    r = libdevice.rsqrt(ms + eps)
    if SAVE_R:
        tl.store(r_ptr + row, r, mask=rmask)
    rb = r[:, None, None]
    t = (row % L)[:, None, None]
    j = tl.arange(0, 32)[None, :, None] * 2 + tl.arange(0, 2)[None, None, :]
    for k in tl.static_range(K):           # chunk k = head k
        a = tl.load(x_ptr + off + k * 128, mask=m, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + woff + k * 128).to(tl.float32)
        yh = (a * rb).to(tl.bfloat16)
        y = (yh.to(tl.float32) * w).to(tl.bfloat16)          # the RMSNorm output (never stored)
        yr, yi = tl.split(tl.reshape(y, (ROWS, 32, 2, 2)))
        orr, oi = _rot(yr, yi, t, j, m, tab_ptr, seq_len, hw, gw, start_frame, CP, C0, C1, False, False, VARIANT)
        tl.store(y_ptr + off + k * 128, tl.reshape(tl.join(orr, oi), (ROWS, 32, 4)), mask=m)


_ROWS = 4
_fwd_launch = _Launcher(_rms_rope_fwd, num_warps=4)


def _forward(x, weight, eps, tab, geom, r):
    L, f, h, w, seq_len, start_frame = geom
    C = x.shape[-1]
    CP = 64
    n = x.numel() // C
    y = torch.empty_like(x)
    _fwd_launch((triton.cdiv(n, _ROWS), 1, 1), (x, weight, tab, y, r if r is not None else y),
                (n, L, seq_len, h * w, w, start_frame, eps, _consts(n, C)[0]),
                (C, CP, CP - 2 * (CP // 3), CP // 3, _ROWS, r is not None, FMA_VARIANT))
    return y


class _RmsRope(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, eps, tab, geom):
        r = torch.empty(x.numel() // x.shape[-1], dtype=torch.float32, device=x.device)
        y = _forward(x, weight, eps, tab, geom, r)
        ctx.save_for_backward(x, weight, r, tab)
        ctx.geom = geom
        return y

    @staticmethod
    def backward(ctx, g):
        # the unfused backward, kernel for kernel: fused_rope's backward, then fused_ew's RMSNorm backward
        # (a fused version that recomputes the rotation in both passes measured no faster: float64-bound)
        x, weight, r, tab = ctx.saved_tensors
        L, f, h, w, seq_len, start_frame = ctx.geom
        B = x.numel() // (L * x.shape[-1])
        g = g.view(B, L, x.shape[-1] // 128, 128)
        if g.stride(-1) != 1:
            g = g.contiguous()
        g = _rope_launch(g, tab, f, h, w, seq_len, start_frame, conj=True, add_zero=True).view(x.shape)
        C = x.shape[-1]
        n = x.numel() // C
        gx = torch.empty_like(x)
        need_gw = ctx.needs_input_grad[1]
        gwp = torch.empty_like(x) if need_gw else gx
        _rms_bwd_launch((triton.cdiv(n, _RMS_ROWS), 1, 1), (g.contiguous(), x, weight, r, gx, gwp),
                        (n, _consts(n, C)[1]), (C, _RMS_ROWS, need_gw))
        gw = gwp.sum_to_size(weight.shape) if need_gw else None
        STATS['bwd'] += 1
        return (gx if ctx.needs_input_grad[0] else None), gw, None, None, None


def _norm_rope(lin, norm, tab, geom):
    if torch.is_grad_enabled() and (lin.requires_grad or norm.weight.requires_grad):
        STATS['grad'] += 1
        return _RmsRope.apply(lin, norm.weight, norm.eps, tab, geom)
    STATS['nograd'] += 1
    return _forward(lin, norm.weight, norm.eps, tab, geom, None)


def qk_norm_rope(attn, x, grid_sizes, freqs, rope_fn, start_frame=0):
    """(rope(norm_q(q(x))), rope(norm_k(k(x))), v(x)) as [B, L, n, d] for WanSelfAttention / CausalWanSelfAttention,
    or None when this case is not covered (the caller runs its own path). rope_fn(t) is the caller's unfused
    rotation, used if the projections come out in a layout the kernel does not take."""
    from .model import WanRMSNorm
    b, s = x.shape[:2]
    n, d = attn.num_heads, attn.head_dim
    nq, nk = attn.norm_q, attn.norm_k
    if not (isinstance(nq, WanRMSNorm) and isinstance(nk, WanRMSNorm) and d == 128 and x.is_cuda
            and b * s >= 16 and not grid_sizes.is_cuda and freqs.is_cuda and freqs.dim() == 2
            and freqs.size(1) == d // 2 and all(p.dtype == torch.bfloat16 and p.is_contiguous() and p.dim() == 1
                                                 for p in (nq.weight, nk.weight))):
        STATS['uncovered'] += 1
        return None
    sizes = grid_sizes.tolist()
    if len(sizes) != b or any(sz != sizes[0] for sz in sizes):
        STATS['uncovered'] += 1
        return None
    f, h, w = sizes[0]
    seq_len = f * h * w
    if seq_len > s or start_frame + f > freqs.size(0) or h > freqs.size(0) or w > freqs.size(0):
        STATS['uncovered'] += 1
        return None
    tab = _table(freqs)
    geom = (s, f, h, w, seq_len, start_frame)

    def one(lin, norm):
        if lin.dtype == torch.bfloat16 and lin.is_contiguous() and lin.shape[-1] == n * d:
            return _norm_rope(lin, norm, tab, geom).view(b, s, n, d)
        STATS['unfused_lin'] += 1
        return rope_fn(norm(lin).view(b, s, n, d))

    q = one(attn.q(x), nq)
    k = one(attn.k(x), nk)
    v = attn.v(x).view(b, s, n, d)
    return q, k, v
