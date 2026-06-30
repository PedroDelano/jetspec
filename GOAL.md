# GOAL — JetSpec via vLLM (vllm-jetspec) on RTX PRO 6000 Blackwell

**Read this file before every step.** _Locked 2026-06-30 (user-confirmed)._

## Target task
Run the **JetSpec vLLM v1 integration** (`JetSpec-project/vllm-jetspec`) on **MATH-500** with:
- **Target:** `Qwen/Qwen3-30B-A3B` (MoE, 30B total / 3B active, bf16)
- **Draft head:** `JetSpec/jetspec-qwen3-30b-a3b`
- **Config:** single-stream **batch=1**, **TP=1** (we have 1 GPU; README example uses TP=4), `--tree-attn-kernel triton`, `--enable-expert-parallel`, budget 128.

Entry script: `examples/offline_inference/jetspec_profiling_math500_tree_budget_bsz_sweep.sh`.

_This is the exact model in the README's vLLM example. **No literal JetSpec metric is published
for any 30B/35B-A3B model** (the engine table is Qwen3-8B only), so the goal is anchored on the
hardware-independent accept-len + a self-referential speedup ratio, NOT a normalized TPS band._

## Metrics & gates

### Primary gate — acceptance length (hardware-INDEPENDENT)
Accept-len depends only on model + draft head + tree config, not the GPU → truest "did JetSpec
reproduce" signal.
- **Target:** MATH-500 accept-len **≥ 7.5** (plausible lossless band ~7.5–10, by analogy to the
  published 8B MATH-500 = 9.56; no exact 30B figure exists).
- **Impl-error floor: accept-len < 3** → tree/draft-head mis-wired, or near 1 = JetSpec not
  engaged / silent AR fallback (RULES §3 violation). Keep looping.

### Secondary gate — end-to-end speedup ratio (self-referential, hardware-fair)
Measure JetSpec TPS vs AR-greedy baseline TPS **on this same card**.
- **Target:** **≥ 5×** on MATH-500 (paper's math family is ~7–9× on Qwen3-8B; treat 7–9× as ideal).
- **Impl-error floor: speedup < 2×** → JetSpec barely helping → our bug. Keep looping.

### Reported (not a gate)
Absolute JetSpec TPS on MATH-500 — recorded for the runbook; no normalized band (no published 30B anchor).

## Correctness criteria
- Loads real `Qwen/Qwen3-30B-A3B` weights + `JetSpec/jetspec-qwen3-30b-a3b` head (pinned revisions).
- Decoding is **lossless** (JetSpec tree path active, accept-len ≫ 1).
- No silent fallback to AR-only, CPU, or a smaller/different model (RULES §3).
- `--tree-attn-kernel triton` (the SM90 CuTe kernel won't run on sm_120 — using triton is an
  explicit, logged choice, not a silent fallback).

## Stop condition
Goal is **MET** when: MATH-500 **accept-len ≥ 7.5** AND **speedup ≥ 5×** AND all Correctness
criteria pass AND all RULES upheld → write `FINAL.md`, stop.
- accept-len **< 3** or speedup **< 2×** → OUR bug → keep looping.
- accept-len in 7.5–10 but speedup between 2× and 5×, traced to a proven host-launch /
  kernel-maturity / FP-precision limit on sm_120 (not a fixable bug) → hardware constraint → met.

## Environment (resolved)
- GPU now **free** (~97 GB, foreign sglang process cleared 2026-06-30).
- Headwinds: nvcc 12.1 predates sm_120 (prefer prebuilt Blackwell wheels; else CUDA ≥12.8);
  vLLM fork may pin an old vLLM lacking Blackwell support (verify after clone).
