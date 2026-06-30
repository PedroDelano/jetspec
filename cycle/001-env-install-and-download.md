# Cycle 001 — env setup, vLLM-fork install, model download

## Goal recap
MATH-500 via vllm-jetspec, Qwen3-30B-A3B + jetspec-qwen3-30b-a3b, batch=1, TP=1, triton tree kernel.
**Stop when:** accept-len ≥ 7.5 AND speedup ≥ 5× AND lossless AND RULES upheld. accept-len<3 or speedup<2× = our bug.

## Attempt
README happy path, adapted for our hardware:
- Base = vLLM **0.11.2.dev278** (Blackwell-aware: CMake CUDA_SUPPORTED_ARCHS includes 12.0/sm_120).
- Our nvcc is 12.1 (can't emit sm_120) → avoid compiling vLLM CUDA: use **cu128 torch + precompiled vLLM .so**, JetSpec tree path is **triton** (JIT, no nvcc) and the fork adds **no JetSpec C++ kernels** (only upstream sm100_mla csrc).
- uv venv (RULE 1). Pin torch/vLLM/flashinfer (RULE 2).
- Download Qwen3-30B-A3B (~60 GB) + jetspec-qwen3-30b-a3b to local dirs (entry script needs local paths).

## Commands run
(to be appended as executed)

## What broke
1. **torch too old.** Installed torch 2.9.0+cu128; `import vllm` →
   `ImportError: cannot import name 'GraphCaptureOutput' from 'torch._dynamo.convert_frame'`.
   `vllm/env_override.py:507` imports it under `if not is_torch_equal_or_newer("2.12.0")`; the
   comment cites pytorch PR 177558 — so it needs torch in [PR177558, 2.12), i.e. **2.10/2.11**.
2. Transient NFS `Stale file handle (os error 116)` left `openai` not installed (re-run fixes).
- Models downloaded fine (16 shards + draft head). vLLM precompiled .so + 134 deps installed.

## Research
- setup.py:608 → base wheel `vllm-0.11.2.dev278+gdbc3d9991`.
- dflash_profiling.py:1920 → `--tree-attn-kernel {triton,optimus}`, triton = "Triton bias-based path", optimus = "fused SM90 paged tree-mask kernel" (won't run on sm_120).
- Entry script runs BOTH `ar` and `dflash` modes → speedup ratio measured directly; summary metric `e2e_throughput_tok_s`.

## Hypothesis
Precompiled-binary install + triton path sidesteps the nvcc-12.1/sm_120 build wall.

## Fix applied
(pending)

## Measured result
did not reach benchmark yet

## GOAL checkpoint
- Within band? n.a. (install phase)
- Correctness pass? n.a.
- Below impl-error threshold? n.a.
- **Goal met?** no

## RULES checkpoint
- Rule 1 uv/pinned: ok (uv venv)
- Rule 2 stack pinned: in progress
- Rule 3 no fallbacks: ok (triton kernel is an explicit logged choice, not silent)
- Rule 4 watchdog ran: n.a. yet

## Progress vs previous cycle
First cycle.
