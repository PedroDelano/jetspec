# Cycle 004 — faster MoE backend + /dev/shm staging

## Goal recap
MATH-500 via vllm-jetspec, Qwen3-30B-A3B + jetspec-qwen3-30b-a3b, batch=1, TP=1, triton tree kernel.
Stop: accept-len ≥ 7.5 (MET, 10.73) AND speedup ≥ 5× (currently 0.76×, FAIL) AND lossless AND RULES.

## Attempt
Throughput is the only failing gate. Tree-verify(128 tok) is 14× costlier/step than AR — MoE on
the forced Triton backend is the suspect for a 30B-A3B MoE. Two changes:
1. **/dev/shm staging** of the model (kill 14-min NFS reload/cell → fast iteration).
2. **Faster MoE backend**: try `flashinfer_trtllm` (precompiled cubins, maybe no nvcc) first;
   if it JIT-fails on sm_120 or isn't faster, install CUDA 12.9 toolkit → `flashinfer_cutlass`.

## Commands run
- cp model + draft head → /dev/shm/models (background)
- JETSPEC_MOE_BACKEND=flashinfer_trtllm bash run_math500.sh 4  (TARGET/DRAFT from /dev/shm)

## What broke (and fixes)
1. Resume skipped cells (same out dir) → added MoE backend to PROFILER_DIR.
2. `CUBLAS_STATUS_INVALID_VALUE` on plain torch GEMM: setting CUDA_HOME=/root/cuda-12.9 made the
   entry script prepend the toolkit lib64 → torch loaded toolkit libcublas/cudart (12.9) over its
   bundled cu128 → mismatch. **Fix:** patched entry script to NOT prepend `$CUDA_HOME/lib64`
   (toolkit is compile-only; torch keeps cu128 runtime). Logged hardware-specific fix.
3. Self-kill: `pkill -f run_math500` matched the launcher's own argv. Use `pkill -f dflash_profiling.py`.
4. flashinfer CUTLASS MoE autotuner emits non-fatal warnings (SM80 generic tactics fail on sm_120:
   "GPU lacks shared memory resources"); it skips them and uses a valid tactic.

## Research
- CUDA 12.9 toolkit (runfile, /root/cuda-12.9) → nvcc supports compute_120; cccl symlinked.
- /dev/shm model staging → model load 846s → ~25s.
- flashinfer fused_moe sm_120 module compiles + caches at /root/.cache/flashinfer/0.6.7/120f/.

## Measured (so far)
- Model load: **25 s** (was 846 s) via /dev/shm. ✅
- **AR baseline with CUTLASS MoE = 143.32 tok/s** (was 118.22 with Triton MoE, +21%).
- **dflash/JetSpec with CUTLASS MoE = 312.61 tok/s, accept-len 10.56** (4-sample smoke, budget 128).
- **Speedup = 312.61 / 143.32 = 2.18×** (was 0.76× with Triton MoE).

## GOAL checkpoint
- Accept-len ≥ 7.5? **YES (10.56)** — algorithm reproduced (HW-independent gate met).
- Speedup ≥ 5×? **NO (2.18×)**, but now **ABOVE the 2× impl-error floor** (no longer "our bug").
- **Goal met?** not yet — speedup in the [2×, 5×) zone. Need to (a) find max speedup via tree-budget
  sweep + more samples, then (b) decide: reach 5×, or declare a proven sm_120 kernel-maturity limit
  (no SM90 optimus tree-attn; CUTLASS MoE falls back to sm80 tactics on sm_120).

## Progress vs previous cycle
Major: speedup 0.76× → 2.18× by enabling fast CUTLASS MoE (CUDA 12.9 nvcc). Model load 846s→25s.

## RULES checkpoint
- R1 uv/pinned ok · R2 stack pinned ok · R3 changing MoE kernel is explicit+logged, lossless · R4 watchdog armed

## Progress vs previous cycle
003 proved algorithm reproduces (accept-len 10.73) but speedup 0.76×; 004 attacks the verify-step cost.
