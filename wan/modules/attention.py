# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import os

import torch

# Use the separately installed official FA3 package. Detect Hopper lazily on the
# first call; SF_ATTN=fa2 keeps the upstream FA2 path.
FLASH_ATTN_3_AVAILABLE = False
if os.environ.get('SF_ATTN', 'fa3') == 'fa3':
    try:
        from flash_attn_3 import flash_attn_interface
        FLASH_ATTN_3_AVAILABLE = True
    except ImportError:
        FLASH_ATTN_3_AVAILABLE = False
_FA3_DEVICES = {}

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False


def _fa3_usable(q, k, v, q_lens, k_lens, lk, dropout_p, window_size, version):
    """FA3 dense path: Hopper, head dim 128, bf16 compute, no dropout/window, and no padding to mask out (k_lens absent
    or a CPU tensor equal to the full key length, so checking it costs no GPU sync)."""
    if not FLASH_ATTN_3_AVAILABLE or version == 2 or dropout_p != 0. or tuple(window_size) != (-1, -1):
        return False
    if q.size(-1) != 128 or k.size(-1) != 128 or v.size(-1) != 128 or v.dtype not in (torch.bfloat16, torch.float32):
        return False
    if q_lens is not None:
        return False
    if k_lens is not None and (k_lens.is_cuda or bool((k_lens != lk).any())):
        return False
    dev = q.device.index if q.device.index is not None else torch.cuda.current_device()
    if dev not in _FA3_DEVICES:
        _FA3_DEVICES[dev] = torch.cuda.get_device_capability(dev)[0] == 9
    return _FA3_DEVICES[dev]


import warnings

__all__ = [
    'flash_attention',
    'attention',
]


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    if dtype == torch.bfloat16 and _fa3_usable(q, k, v, q_lens, k_lens, lk, dropout_p, window_size, version):
        # same casts as the varlen path below, dense [B, L, N, C] layout, no length tensors to build
        q, k, v = half(q), half(k), half(v)
        q = q.to(v.dtype)
        k = k.to(v.dtype)
        if q_scale is not None:
            q = q * q_scale
        x = flash_attn_interface.flash_attn_func(
            q=q,
            k=k,
            v=v,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic or torch.are_deterministic_algorithms_enabled())
        return x.type(out_dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        if torch.cuda.is_current_stream_capturing():  # SF_CUDAGRAPH: same lengths, built on the device (no H2D copy)
            q_lens = torch.full([b], lq, dtype=torch.int32, device=q.device)
        else:
            q_lens = torch.tensor(
                [lq] * b, dtype=torch.int32).to(
                    device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        if torch.cuda.is_current_stream_capturing():
            k_lens = torch.full([b], lk, dtype=torch.int32, device=k.device)
        else:
            k_lens = torch.tensor(
                [lk] * b, dtype=torch.int32).to(
                    device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale


    # apply attention
    if False:  # Keep padded inputs on the existing FA2 path
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
        return out
