from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from functools import partial
import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy, StateDictType
from torch.distributed.fsdp.api import CPUOffload
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy


def fsdp_state_dict(model):
    fsdp_fullstate_save_policy = FullStateDictConfig(
        offload_to_cpu=True, rank0_only=True
    )
    with FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, fsdp_fullstate_save_policy
    ):
        checkpoint = model.state_dict()

    return checkpoint


def fsdp_wrap(module, sharding_strategy="full", mixed_precision=False, wrap_strategy="size", min_num_params=int(5e7), transformer_module=None, cpu_offload=False):
    if mixed_precision:
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
            cast_forward_inputs=False
        )
    else:
        mixed_precision_policy = None

    if wrap_strategy == "transformer":
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_module
        )
    elif wrap_strategy == "size":
        auto_wrap_policy = partial(
            size_based_auto_wrap_policy,
            min_num_params=min_num_params
        )
    else:
        raise ValueError(f"Invalid wrap strategy: {wrap_strategy}")

    os.environ["NCCL_CROSS_NIC"] = "1"

    sharding_strategy = {
        "full": ShardingStrategy.FULL_SHARD,
        "hybrid_full": ShardingStrategy.HYBRID_SHARD,
        "hybrid_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,
        "no_shard": ShardingStrategy.NO_SHARD,
    }[sharding_strategy]

    module = FSDP(
        module,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=sharding_strategy,
        mixed_precision=mixed_precision_policy,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        use_orig_params=True,
        cpu_offload=CPUOffload(offload_params=cpu_offload),
        sync_module_states=False  # Load ckpt on rank 0 and sync to other ranks
    )
    return module


def barrier():
    if dist.is_initialized():
        dist.barrier()


def launch_distributed_job(backend: str = "nccl"):
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    host = os.environ["MASTER_ADDR"]
    port = int(os.environ["MASTER_PORT"])

    if ":" in host:  # IPv6
        init_method = f"tcp://[{host}]:{port}"
    else:  # IPv4
        init_method = f"tcp://{host}:{port}"
    dist.init_process_group(rank=rank, world_size=world_size, backend=backend,
                            init_method=init_method, timeout=timedelta(minutes=30))
    torch.cuda.set_device(local_rank)


# SF_EMA_FAST=1: same EMA arithmetic, without the per-parameter synchronous pageable D2H copies and the single CPU
# thread (torchrun sets OMP_NUM_THREADS=1). Parameters are staged into one pinned buffer with async copies, and the
# CPU update runs on a thread pool in chunks whose boundaries are multiples of _EMA_CHUNK. ATen's CPU loop works in
# SIMD blocks of 2 x Vec::size() elements from the start of each call and finishes the remainder with a scalar loop,
# so with block-aligned chunk boundaries every element takes the same (vector or scalar) path as one call over the
# whole tensor: the result is bitwise the same as the plain loop in update().
_EMA_FAST = os.environ.get("SF_EMA_FAST", "0") == "1"
_EMA_THREADS = int(os.environ.get("SF_EMA_THREADS", "16"))
# SF_EMA_ASYNC=1 (with SF_EMA_FAST): update() returns as soon as the copies are queued; the CPU update (the same chunked
# jobs) runs on a background thread while the next iteration launches its GPU work, instead of the GPU idling through
# it. Every reader of the shadow -- state_dict(), load_state_dict(), copy_to(), the next update() -- waits for it first,
# so whatever is read is bitwise the shadow of the synchronous update.
_EMA_ASYNC = os.environ.get("SF_EMA_ASYNC", "0") == "1"
_EMA_CHUNK = 1 << 20   # elements; a multiple of every SIMD block width (AVX-512: 2 x 16 floats)


class EMA_FSDP:
    def __init__(self, fsdp_module: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self._init_shadow(fsdp_module)

    @torch.no_grad()
    def _init_shadow(self, fsdp_module):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        with FSDP.summon_full_params(fsdp_module, writeback=False):
            for n, p in fsdp_module.module.named_parameters():
                self.shadow[n] = p.detach().clone().float().cpu()

    @torch.no_grad()
    def update(self, fsdp_module):
        self._join()
        d = self.decay
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        with FSDP.summon_full_params(fsdp_module, writeback=False):
            # bitwise argument needs one intra-op thread (torchrun default), as the plain loop then runs each tensor
            # in a single call; with more threads ATen would split both loops at different, unaligned points
            if _EMA_FAST and torch.get_num_threads() == 1:
                self._update_staged(fsdp_module.module.named_parameters(), d)
                return
            for n, p in fsdp_module.module.named_parameters():
                self.shadow[n].mul_(d).add_(p.detach().float().cpu(), alpha=1. - d)

    @torch.no_grad()
    def _update_staged(self, named_parameters, d):
        """update() for SF_EMA_FAST (see _EMA_FAST): shadow = shadow * d + float(p) * (1 - d), bitwise as update()."""
        params = [(n, p.detach()) for n, p in named_parameters]
        total = sum(p.numel() for _, p in params)
        if getattr(self, "_staging", None) is None or self._staging.numel() < total:
            self._staging = torch.empty(total, dtype=torch.float32, pin_memory=True)
        if getattr(self, "_pool", None) is None:
            self._pool = ThreadPoolExecutor(max_workers=_EMA_THREADS, thread_name_prefix="ema")
        stream = torch.cuda.current_stream()
        staged, offset = [], 0
        for n, p in params:
            buf = self._staging[offset:offset + p.numel()]
            buf.copy_(p.float().reshape(-1), non_blocking=True)
            done = torch.cuda.Event()
            done.record(stream)
            staged.append((n, buf, done))
            offset += p.numel()

        alpha = 1. - d

        def axpby(shadow, x):
            shadow.mul_(d).add_(x, alpha=alpha)

        def apply():
            jobs = []
            for n, buf, done in staged:
                done.synchronize()          # this parameter has landed; later copies keep streaming meanwhile
                shadow = self.shadow[n].view(-1)
                for a in range(0, shadow.numel(), _EMA_CHUNK):
                    jobs.append(self._pool.submit(axpby, shadow[a:a + _EMA_CHUNK], buf[a:a + _EMA_CHUNK]))
            for job in jobs:
                job.result()

        if _EMA_ASYNC:
            if getattr(self, "_background", None) is None:
                self._background = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ema-async")
            self._pending = self._background.submit(apply)
        else:
            apply()

    def _join(self):
        """Wait for a background (SF_EMA_ASYNC) update: it owns the shadow and the staging buffer until it finishes."""
        pending = getattr(self, "_pending", None)
        if pending is not None:
            self._pending = None
            pending.result()

    # Optional helpers ---------------------------------------------------
    def state_dict(self):
        self._join()
        return self.shadow            # picklable

    def load_state_dict(self, sd):
        self._join()
        self.shadow = {k: v.clone() for k, v in sd.items()}

    def copy_to(self, fsdp_module):
        # load EMA weights into an (unwrapped) copy of the generator
        self._join()
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        with FSDP.summon_full_params(fsdp_module, writeback=True):
            for n, p in fsdp_module.module.named_parameters():
                if n in self.shadow:
                    p.data.copy_(self.shadow[n].to(p.dtype, device=p.device))
