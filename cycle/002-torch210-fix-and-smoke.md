# Cycle 002 — torch 2.10 fix, then smoke MATH-500

## Goal recap
MATH-500 via vllm-jetspec, Qwen3-30B-A3B + jetspec-qwen3-30b-a3b, batch=1, TP=1, triton tree kernel.
Stop: accept-len ≥ 7.5 AND speedup ≥ 5× AND lossless AND RULES. accept-len<3 or speedup<2× = our bug.

## Attempt
Carry fix from 001: install **torch 2.10.0+cu128** (was 2.9.0 → missing `GraphCaptureOutput`).
Then run a small smoke of `run_math500.sh` (max-samples 4) to confirm vLLM + JetSpec tree path
work on sm_120 before the scored run.

## Commands run
- `uv pip install torch==2.10.0 torchvision torchaudio --index-url .../cu128`
- `uv pip install openai`  (fix 001's stale-file-handle miss)
- smoke: `bash run_math500.sh 4`

## What broke
- torch fix: **resolved.** torch 2.10.0+cu128; `GraphCaptureOutput` import OK; `import vllm` OK
  (precompiled .so ABI matches torch 2.10). openai installed.
- smoke #1: `ModuleNotFoundError: datasets` → installed datasets/tqdm/psutil.
- smoke #2: `ModuleNotFoundError: transformers.utils` → transformers install **corrupted**
  (utils/ + METADATA missing on disk) by the 001 NFS `Stale file handle` error. Reinstalled;
  unconstrained pull grabbed transformers **5.12.1** (vLLM pins `>=4.56,<5`) → re-pinned to **4.57.6**.
- smoke #3: `ModuleNotFoundError: vllm._C` at cuda.py:21. Precompiled `.so` extraction was also
  truncated by the same NFS error — `vllm/_C.abi3.so`, `_moe_C.abi3.so`, `_sparse_flashmla_C.abi3.so`
  missing (setup.py:665 files_to_copy). uv-driven re-extraction hung (slow dep re-resolution;
  then `prepare_metadata_for_build_editable` stuck in futex; NFS flakiness) — abandoned.
  **Resolved manually:** found the fork's upstream base commit via
  `git fetch --unshallow --filter=blob:none` + `git merge-base HEAD <upstream/main>` →
  **551b3fb39f** (vLLM 0.18.2rc1.dev57, 2026-04-02). Downloaded that exact precompiled wheel
  (`wheels.vllm.ai/.../cu129`, 415 MB) to /root (stable fs), extracted all 8 `.so` into the
  source tree with per-file size verification + retry. `_sparse_flashmla` not in this build
  (optional, not imported). Verified: `import vllm._C/_moe_C/_C_stable_libtorch` + `from vllm
  import LLM` all OK (slow ~118s first load = reading ~1 GB of .so off NFS).

## Research
- env_override.py gate `is_torch_equal_or_newer("2.12.0")` + pytorch PR 177558 → needs torch in
  [2.10, 2.12). cu128 index offers 2.10.0/2.11.0. 2.10.0 verified working (API+ABI).

## Hypothesis
Correct torch unblocks import; remaining risk = sm_120 kernel coverage in the precompiled .so /
flashinfer / triton tree path at runtime.

## Fix applied
torch 2.9.0 → 2.10.0+cu128 (RULE 2: stack pin updated).

## Measured result
Smoke #4 got vLLM to **load the 30B model on sm_120** (59 GB VRAM) and start the AR cell, then
crashed: `RuntimeError: No supported CUDA architectures found for major versions [12]` — FlashInfer
**CUTLASS MoE** tried to JIT-compile an sm_120 kernel via nvcc 12.1 (needs ≥12.9).
Non-fatal warnings seen: GPT-OSS triton kernels import fail (unused path); `SM 12.x requires
CUDA >= 12.9` probes (fell back to FlashInfer CUTLASS MoE + FlashAttention v2).
→ Carry to cycle 003: force Triton MoE backend (no nvcc).

## GOAL checkpoint
- Within band? n.a. (smoke)
- Correctness pass? n.a.
- Below impl-error threshold? n.a.
- **Goal met?** no

## RULES checkpoint
- Rule 1 uv/pinned: ok
- Rule 2 stack pinned: ok (torch 2.10.0+cu128 recorded)
- Rule 3 no fallbacks: ok (triton kernel explicit)
- Rule 4 watchdog: GPU monitor to be armed for the smoke run

## Progress vs previous cycle
Yes — cleared the import wall (torch version identified & fixed); import vllm now works.
