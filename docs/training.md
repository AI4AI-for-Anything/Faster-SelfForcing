# Training and reproduction

The optimizations are environment switches on the upstream trainer. Every switch is off unless set; with none set, the
code runs the upstream path except for attention, which uses the installed FlashAttention-3 on Hopper GPUs (see below).
The measured configuration is [`scripts/configs/faster.env`](../scripts/configs/faster.env).

## Environment

The benchmark ran with **PyTorch 2.7.1 + CUDA 12.8** on Hopper GPUs (compute capability 9.0), Python 3.10:

```bash
conda create -n self_forcing python=3.10 -y
conda activate self_forcing
pip install torch==2.7.1 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
python setup.py develop
```

Install FA3 separately using the [official FlashAttention-3 instructions](https://github.com/Dao-AILab/flash-attention#flashattention-3-beta-release). In a separate source directory:

```bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/hopper
python setup.py install
python -c 'from flash_attn_3 import flash_attn_interface'
```

The official FA3 instructions require Hopper hardware (H100/H800) and CUDA 12.3 or newer, and recommend CUDA 12.8. Keep FA2 installed for fallback. This repository uses FA3 for supported Hopper inputs with head dimension 128 and BF16 computation; other inputs use FA2. Set `SF_ATTN=fa2` to force FA2.

The reported 2.08× benchmark used a custom FA3 build from revision `b54df16` with PyTorch 2.7.1+cu128. That binary is no longer distributed here. The official installation above has not yet been benchmarked in this project; its performance and numerical results may differ from the recorded run.

## Public data and weights

The trainer is data-free: it needs the text prompts, the ODE-initialized generator and the two Wan2.1 models (1.3B
generator / critic, 14B teacher). Download them into the repository root:

```bash
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir wan_models/Wan2.1-T2V-1.3B
huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir wan_models/Wan2.1-T2V-14B
huggingface-cli download gdhe17/Self-Forcing checkpoints/ode_init.pt --local-dir .
huggingface-cli download gdhe17/Self-Forcing vidprom_filtered_extended.txt --local-dir prompts
```

## Launch

Source the switches, then use the upstream command. The upstream recipe trains for 600 iterations on 64 GPUs, one
video per GPU:

```bash
source scripts/configs/faster.env
torchrun --nnodes=8 --nproc_per_node=8 --rdzv_id=5235 \
  --rdzv_backend=c10d --rdzv_endpoint $MASTER_ADDR \
  train.py --config_path configs/self_forcing_dmd.yaml \
  --logdir logs/faster_self_forcing_dmd --disable-wandb
```

At exit, the switches that have counters print them (for example `[SF_SAC_SAVE]`, the fused-kernel call counts and
the T5 gather count). A zero count means that path never ran; check them before trusting a timing.

## Switches

| Switch | What it does | Numerics |
|---|---|---|
| FlashAttention-3 (default) | Installed FA3 dense attention on Hopper; `SF_ATTN=fa2` restores FA2 | Not bit-identical to FA2 |
| `SF_EMA_FAST` | Same EMA arithmetic without per-parameter synchronous copies | Bit-identical |
| `SF_EMA_ASYNC` | EMA CPU update in a background thread, joined before every read | Bit-identical |
| `SF_NO_SYNC` | Removes host synchronizations from the rollout | Bit-identical |
| `SF_ROPE_FUSED` | Fused rotary embedding | Bit-identical |
| `SF_SAC` | Selective activation checkpointing: the recompute reuses attention outputs | Bit-identical |
| `SF_SAC_SAVE=ffn2,o,co` | Also keeps these linear outputs (bidirectional models by default) | Bit-identical |
| `SF_COMPILE`, `SF_COMPILE_MIN_TOKENS` | Compiled pointwise chains; short rollout calls stay eager | Bit-identical |
| `SF_FSDP_HOLD` | Generator all-gathered once per rollout and reused by the no-grad forwards | Bit-identical |
| `SF_XATTN_CACHE` | Cross-attention K/V computed once per rollout | Bit-identical |
| `SF_CUDAGRAPH` | CUDA graphs for the no-grad KV-cache forwards (needs `SF_NO_SYNC` and `SF_FSDP_HOLD`) | Bit-identical |
| `SF_FUSE_EW` | Triton RMSNorm (forward and backward) and rollout-size modulate / gated residual | Bit-identical |
| `SF_EW_LONG` | Triton modulate / gated residual at every size | Bit-identical |
| `SF_FUSE_QKROPE` | q/k RMSNorm and RoPE in one forward kernel | Bit-identical |
| `SF_FUSE_LNMOD` | LayerNorm (bit-exact with ATen) fused with modulate | Bit-identical |
| `SF_FROZEN_BF16` | Frozen teacher and text encoder keep BF16 parameters; no per-forward FP32→BF16 casts | Bit-identical |
| `SF_T5_RESIDENT` | Text encoder all-gathered once and kept | Bit-identical |

The numerics column describes the recorded tests of the historical configuration. "Bit-identical" means that, with the setting enabled and FA2 attention, all recorded tensors (losses, outputs, gradients, updates, optimizer state and buffers) matched the reference exactly. With FA3 enabled, the final configuration instead passed the short-trajectory statistical check described in the README. A separately installed FA3 build requires its own validation.

## Memory

At one video per GPU on two GPUs, peak allocated memory per GPU was **93.55 GiB (reference) → 83.57 GiB (optimized)**,
and peak reserved memory **128.4 → 125.8 GiB**, on GPUs with 140 GiB of memory. The upstream config uses hybrid
sharding, which shards parameters within a node, so the footprint is smaller with eight GPUs per node; an 80 GB
configuration has not been measured.
`SF_SAC_SAVE` and `SF_CUDAGRAPH` trade memory for speed; leave them unset first if memory is short.

## Benchmark protocol

- Workload: `configs/self_forcing_dmd.yaml` unchanged except `ema_start_step: 0` (EMA is active for 400 of the 600
  upstream iterations, so the timing measures that steady state). One video per GPU, 21 latent frames at 60×104,
  critic : generator updates 5 : 1.
- An iteration is one critic optimizer step; every fifth iteration also has a generator step. The timer runs the
  repository's real `train.py` and skips 10 warm-up iterations, then times two windows of 10 iterations.
- Reference → candidate → reference in independent processes. Each window is checked on its own; a window that
  deviates from the frozen reference invalidates the run.
- Validation: structure checks, exact tensor comparisons, and statistical comparisons of 100-step training runs with matched seeds. The acceptance threshold for each metric is twice the reference standard deviation across seeds, set before testing optimized configurations.

The historical FA3 comparison recorded 12,379 exact matches among 40,490 tensors; the final configuration did not match the reference bit for bit. Its statistical check used three matched random seeds and measured critic and generator losses and gradient norms at steps 50 and 100. The README reports mean differences normalized by the reference standard deviation across seeds. These measurements concern short training trajectories, not generated-video quality.
