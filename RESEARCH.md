# RESEARCH — JetSpec (hao-ai-lab)

_Compiled 2026-06-30._

## What it is
JetSpec = **causal parallel tree drafting** for lossless LLM speculative decoding. A
causal-parallel draft head reads fused hidden states from the frozen target and emits
per-depth logits in **one parallel pass**; tree construction spends a draft budget over
high-probability branches; the target verifies the whole token tree in one tree-masked
forward. Accepted path follows the target's own logits → **lossless by construction**.

- GitHub: https://github.com/hao-ai-lab/JetSpec
- Paper: https://arxiv.org/abs/2606.18394 (Hu et al., 2026)
- Blog: https://haoailab.com/blogs/parallel-tree-decoding/
- HF org (draft heads): https://huggingface.co/JetSpec
- vLLM fork: https://github.com/JetSpec-project/vllm-jetspec

## Claimed metrics (Qwen3-8B, single-stream batch=1, bf16)
**Engine results — NVIDIA B200** (README table):

| dataset | TPS | accept_len |
|---|---:|---:|
| MATH-500 | **1150 tok/s** | **9.56** |
| GSM8K | 984 | 7.94 |
| HumanEval | 867 | 6.92 |
| AIME25 | 849 | 8.79 |
| MBPP | 789 | 7.61 |
| LiveCodeBench | 684 | 7.66 |
| MT-Bench | 545 | 4.88 |

**End-to-end speedup vs AR greedy (Qwen3-8B)**: MATH-500 **9.64×**, GSM8K 7.82×, AIME25 8.78×,
HumanEval 7.12×, MBPP 6.73×, LCB 7.67×, MT-Bench 4.58×. (Paper/blog report ratios on **H100**;
README headline TPS reported on **B200**.)

README notes the engine numbers "closely align with the vLLM v1 integration."

## Two execution paths (this repo)
1. `jetspec/core/` — lightweight HF-transformers reference (`LLM`, triton tree attention).
2. `jetspec/inference_engine/` — optimized engine: paged KV, Triton tree attention, CUDA graphs. Produces the engine-results table above.

## vLLM path (what the user wants) — separate repo `vllm-jetspec`
- vLLM **v1** fork with JetSpec support.
- README runnable example: target **Qwen3-30B-A3B**, **TP=4**, `--tree-attn-kernel triton`,
  `--enable-expert-parallel`, `--disable-cascade-attn`, `--cudagraph-mode default`,
  budget 128, MATH-500, `GPU_MEMORY_UTILIZATION=0.90`.
- Entry script: `examples/offline_inference/jetspec_profiling_math500_tree_budget_bsz_sweep.sh`
  (HumanEval: `jetspec_profiling_humaneval_tree_unit_kvlayout.sh`).
- ✅ The runnable example uses **`--tree-attn-kernel triton`**, NOT the SM90 CuTe kernel.

## Known gotchas / risks for OUR hardware (sm_120 Blackwell, nvcc 12.1)
1. **SM90 CuTe DSL kernel:** the paper/press describe a "custom SM90 (Hopper) paged
   FlashAttention kernel via NVIDIA CuTe DSL" for tree verification. SM90 ≠ sm_120 — that
   kernel likely won't run on Blackwell. **Mitigation:** the README's runnable example uses the
   **triton** tree-attn kernel, which is arch-portable. Plan to use `--tree-attn-kernel triton`.
2. **nvcc 12.1 vs sm_120:** building vLLM/custom CUDA kernels from source for Blackwell needs
   CUDA ≥ 12.8. Prefer prebuilt Blackwell wheels; else install a newer CUDA toolkit.
3. **vLLM fork version pin:** the fork may pin an older vLLM lacking Blackwell support
   (`VLLM_FORK_DIR` confusingly points the example at the JetSpec dir). To verify after clone.
4. **Single GPU:** example uses TP=4 + expert-parallel for the 30B-A3B MoE. We have 1 GPU →
   run **TP=1**. The 30B-A3B MoE at bf16 (~60 GB) fits in 96 GB but is heavy; cleanest
   apples-to-apples is **Qwen3-8B** (the model the 1150/9.56 numbers are quoted for).
5. **VRAM occupied:** 83 GB held by a foreign sglang process at detection — must be freed.
6. **Host-launch-bound batch-1:** README states the batch-1 tree round is host-launch-bound, so
   absolute TPS won't scale purely with GPU bandwidth → accept-len is the cleaner correctness signal.

## Capability numbers for normalization
- B200 HBM3e: **~8,000 GB/s** (8 TB/s).
- RTX PRO 6000 Blackwell: **1,792 GB/s** (see HARDWARE.md).
- Bandwidth ratio ours/B200 = 1792 / 8000 = **0.224**.
