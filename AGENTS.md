# Repository instructions

- Keep the upstream workload: `configs/self_forcing_dmd.yaml`, one video per GPU, critic : generator updates 5 : 1. The 2.08× headline is a two-GPU per-iteration benchmark; do not present it as an end-to-end training speedup.
- Keep optimization switches off by default. In the recorded FA2 tests, the optimization settings matched the baseline on recorded tensors; the historical FA3 configuration passed the short-trajectory statistical check. Do not extend those results to untested builds or claim lossless acceleration.
- A full 600-iteration run and generated-video quality have not been validated. Do not fill them in with paper numbers. Put new experiments in a new dated directory with source, dependency and hardware provenance.
- Install FA3 from the official FlashAttention repository. The recorded benchmark used a custom build; validate a separately installed build before attributing the historical speedup or numerical verdict to it.
- Check the exit counters (fused kernels, `[SF_SAC_SAVE]`, T5 gathers) before trusting a timing: a path that silently fell back looks the same as one that did not help.
