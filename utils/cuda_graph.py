"""SF_CUDAGRAPH=1: CUDA-graph replay of the causal generator's no-grad KV-cache forwards (the Self-Forcing rollout).

The rollout is CPU-bound: every 3-frame forward of the causal 1.3B model launches ~2k small kernels. In no-grad mode
the forward for a given block position (current_start) always runs the same kernels on the same buffers, so it is
captured once and replayed afterwards:
  - the hook sits in CausalWanModel.forward, inside the FSDP unit: FSDP still runs its pre/post forward around every
    call, so the top-level forward count and the parameter sharding are untouched;
  - parameters: the graphs read static copies. FSDP places the unsharded parameters at a new address whenever it
    all-gathers, but their values only change at optimizer steps, which happen outside rollouts. Under SF_FSDP_HOLD the
    copy is refreshed once per rollout (utils/fsdp_hold.EPOCH); without it, before every replay;
  - latents, timesteps and prompt embeddings are copied into static buffers before a replay; the self-attention KV
    cache is used in place (the pipeline keeps it at fixed addresses under SF_CUDAGRAPH);
  - cross-attention cache: under FSDP without SF_XATTN_CACHE the root copies the cache dicts, is_init never sticks and
    every forward recomputes the context K/V ('compute' graphs); with SF_XATTN_CACHE the slots keep the K/V of the
    rollout's first no-grad forward ('cached' graphs read static copies, refreshed when the slot tensors change). The
    python side effects (is_init and the K/V entries, the SF_NO_SYNC host index) are applied by hand after a replay;
  - nothing random runs inside a graph (the rollout's randn calls stay outside).
Grad-enabled forwards (the backpropagated exit step) run eagerly. Needs SF_NO_SYNC=1 (no GPU->CPU sync in the forward).
"""
import os
from collections import Counter

import torch

from utils import fsdp_hold


def enabled():
    return os.environ.get('SF_CUDAGRAPH') == '1'


class CausalForwardGraphs:
    def __init__(self, model):
        self.model = model          # CausalWanModel; called from its forward, inside the FSDP unit
        self.entries = {}
        self.static = {}
        self.pool = None
        self.stream = None
        self.sparams = None
        self.params_epoch = None    # fsdp_hold.EPOCH the static parameters were copied in
        self.cross_src = None       # the cached cross-attention K/V last copied into the static buffers
        self.disabled = False
        self.stats = Counter()
        self.param_ptrs = []        # held parameter address at each refresh (diagnostics)

    def eligible(self, kv_cache, crossattn_cache):
        return (enabled() and not self.disabled and not torch.is_grad_enabled()
                and kv_cache is not None and crossattn_cache is not None
                and getattr(self.model, 'local_attn_size', -1) == -1
                and len({bool(c.get('is_init')) for c in crossattn_cache}) == 1
                and all(c.get('host_index') is not None for c in kv_cache)
                and not torch.cuda.is_current_stream_capturing())

    @staticmethod
    def _cache_ptrs(kv_cache):
        return tuple(c[n].data_ptr() for c in kv_cache for n in ('k', 'v', 'global_end_index', 'local_end_index'))

    def _refresh_params(self):
        """Copy the (unsharded, inside the FSDP forward) parameters into the static buffers when they may have changed."""
        epoch = fsdp_hold.EPOCH if fsdp_hold.ACTIVE else None
        if self.sparams is not None and epoch is not None and epoch == self.params_epoch:
            return
        params = list(self.model.parameters())
        if self.sparams is None:
            self.sparams = [torch.empty_like(p) for p in params]
        torch._foreach_copy_(self.sparams, params)
        self.params_epoch = epoch
        self.stats['param_refresh'] += 1
        ptr = params[0].data_ptr()
        if self.param_ptrs and ptr != self.param_ptrs[-1]:
            self.stats['param_moved'] += 1
            if self.stats['param_moved'] <= 5:
                print(f'[SF_CUDAGRAPH] unsharded parameters moved since the last refresh ({dict(self.stats)})', flush=True)
        self.param_ptrs = (self.param_ptrs + [ptr])[-64:]

    def _refresh_cross(self, e, crossattn_cache):
        src = [c[n] for c in crossattn_cache for n in ('k', 'v')]
        if self.cross_src is not None and len(src) == len(self.cross_src) and all(
                a is b for a, b in zip(src, self.cross_src)):
            return
        torch._foreach_copy_(e['cross_dst'], src)
        self.cross_src = src
        self.stats['cross_refresh'] += 1

    def __call__(self, x, t, context, seq_len, kv_cache, crossattn_cache, current_start, cache_start):
        """x: [B, C, F, H, W] (the caller's permuted latents), t: [B, F], context: [B, L, C]."""
        host = [(c['host_index'][0], c['host_index'][1]) for c in kv_cache]
        cached = bool(crossattn_cache[0].get('is_init'))
        # x is the caller's [B, F, C, H, W] latents permuted; a dense base is one layout whatever the (meaningless)
        # stride of its size-1 batch dim (a slice of the noise tensor vs a fresh add_noise result)
        layout = 'dense' if x.permute(0, 2, 1, 3, 4).is_contiguous() else tuple(x.stride())
        key = (tuple(x.shape), layout, x.dtype, tuple(t.shape), t.dtype, tuple(context.shape), context.dtype,
               current_start, cache_start, tuple(h[0] - h[1] for h in host), cached)
        caches = self._cache_ptrs(kv_cache)
        e = self.entries.get(key)
        if e is not None and e['caches'] != caches:
            # the KV cache moved (the pipeline re-allocated it): drop every graph and the pool they share
            self.entries.clear()
            self.pool = None
            self.cross_src = None
            e = None
            self.stats['reset'] += 1
            print(f'[SF_CUDAGRAPH] KV cache moved: graphs dropped ({dict(self.stats)})', flush=True)
        self._refresh_params()
        if e is None:
            e = self._capture(key, x, t, context, seq_len, kv_cache, crossattn_cache, current_start, cache_start,
                              caches, cached)
            if e is None:
                return None
        else:
            torch._foreach_copy_(e['dst'], [x, t, context])
            if cached:
                self._refresh_cross(e, crossattn_cache)
        e['graph'].replay()
        for c, h in zip(kv_cache, e['host_after']):
            c['host_index'][0], c['host_index'][1] = h
        if not cached:                                  # what the eager forward leaves in the caller's slots
            for c, kv in zip(crossattn_cache, e['cross_out']):
                c['is_init'] = True
                c['k'], c['v'] = kv[0].clone(), kv[1].clone()
        self.stats['replay'] += 1
        if self.stats['replay'] % 2000 == 0:
            print(f'[SF_CUDAGRAPH] {dict(self.stats)}', flush=True)
        return e['out'].clone()

    def _capture(self, key, x, t, context, seq_len, kv_cache, crossattn_cache, current_start, cache_start, caches,
                 cached):
        dev = x.device
        # static inputs are shared by all graphs of the same shapes (replays are sequential and refill them)
        skey = (key[:7], cached, tuple((tuple(c['k'].shape), c['k'].dtype) for c in crossattn_cache) if cached else None)
        if skey not in self.static:
            if key[1] == 'dense':
                B, C, F, H, W = x.shape
                sx = torch.empty((B, F, C, H, W), dtype=x.dtype, device=dev).permute(0, 2, 1, 3, 4)
            else:
                sx = torch.empty_strided(x.size(), x.stride(), dtype=x.dtype, device=dev)
            st, sc = torch.empty_like(t), torch.empty_like(context)
            scross = [{'k': torch.empty_like(c['k']), 'v': torch.empty_like(c['v'])} for c in crossattn_cache] \
                if cached else None
            cross_dst = [d[n] for d in scross for n in ('k', 'v')] if cached else []
            self.static[skey] = (sx, st, sc, scross, cross_dst)
        sx, st, sc, scross, cross_dst = self.static[skey]
        torch._foreach_copy_([sx, st, sc], [x, t, context])
        if cached:
            torch._foreach_copy_(cross_dst, [c[n] for c in crossattn_cache for n in ('k', 'v')])
            self.cross_src = [c[n] for c in crossattn_cache for n in ('k', 'v')]
        host0 = [(c['host_index'][0], c['host_index'][1]) for c in kv_cache]
        n_blocks = len(crossattn_cache)
        holder = {}
        params = list(self.model.parameters())

        def run():
            for c, h in zip(kv_cache, host0):
                c['host_index'][0], c['host_index'][1] = h
            if cached:
                cross = [{'k': d['k'], 'v': d['v'], 'is_init': True} for d in scross]
            else:
                cross = [{'k': None, 'v': None, 'is_init': False} for _ in range(n_blocks)]
            holder['cross'] = cross
            return self.model._forward_inference(sx, t=st, context=sc, seq_len=seq_len, kv_cache=kv_cache,
                                                 crossattn_cache=cross, current_start=current_start,
                                                 cache_start=cache_start)

        cur = torch.cuda.current_stream(dev)
        if self.stream is None:                         # one capture stream: blocks cached per stream stay reusable
            self.stream = torch.cuda.Stream(device=dev)
        s = self.stream
        s.wait_stream(cur)
        g = torch.cuda.CUDAGraph()
        live = [p.data for p in params]
        try:
            for p, a in zip(params, self.sparams):      # the graph reads the static parameter buffers
                p.data = a
            with torch.cuda.stream(s):
                run()                                   # warm-up on the capture stream; rewrites the same cache slots
                if self.pool is None:
                    self.pool = torch.cuda.graph_pool_handle()
                g.capture_begin(pool=self.pool, capture_error_mode='thread_local')
                try:
                    out = run()
                finally:
                    g.capture_end()
        except Exception as err:                        # not capturable: stay eager from now on
            cur.wait_stream(s)
            self.disabled = True
            self.stats['capture_failed'] += 1
            print(f'[SF_CUDAGRAPH] capture failed at current_start={current_start}, graphs disabled: '
                  f'{type(err).__name__}: {err}', flush=True)
            for c, h in zip(kv_cache, host0):
                c['host_index'][0], c['host_index'][1] = h
            return None
        finally:
            for p, d in zip(params, live):
                p.data = d
        cur.wait_stream(s)
        e = {'graph': g, 'dst': [sx, st, sc], 'cross_dst': cross_dst, 'out': out,
             'host_after': [(c['host_index'][0], c['host_index'][1]) for c in kv_cache], 'caches': caches,
             'cross_out': None if cached else [(c['k'], c['v']) for c in holder['cross']]}
        for c, h in zip(kv_cache, host0):                # the caller's replay applies the forward once more
            c['host_index'][0], c['host_index'][1] = h
        self.entries[key] = e
        self.stats['capture'] += 1
        print(f'[SF_CUDAGRAPH] captured current_start={current_start} cross={"cached" if cached else "compute"} '
              f'({dict(self.stats)})', flush=True)
        return e
