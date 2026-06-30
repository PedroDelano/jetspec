#!/usr/bin/env bash
# Adapted JetSpec vLLM MATH-500 run for RTX PRO 6000 Blackwell (sm_120), single GPU.
# Usage: run_math500.sh [MAX_SAMPLES]   (default 16 = smoke; pass 0 for full MATH-500)
set -euo pipefail

source /workspace/jetspec-v2/.venv/bin/activate

# CUDA 12.9 toolkit (nvcc supports sm_120) so FlashInfer can JIT CUTLASS MoE for Blackwell.
export CUDA_HOME=/root/cuda-12.9
export PATH="$CUDA_HOME/bin:$PATH"
# MoE backend: flashinfer_cutlass (fast, JIT via nvcc 12.9) by default now; override via env.
export JETSPEC_MOE_BACKEND="${JETSPEC_MOE_BACKEND:-flashinfer_cutlass}"

export VLLM_FORK_DIR=/workspace/jetspec-v2/vllm-jetspec
# Models staged in /dev/shm (tmpfs/RAM) for fast loading; fall back to NFS if absent.
export TARGET_MODEL=/dev/shm/models/Qwen3-30B-A3B
export DRAFT_MODEL=/dev/shm/models/jetspec-qwen3-30b-a3b
[ -d "$TARGET_MODEL" ] || export TARGET_MODEL=/workspace/jetspec-v2/models/Qwen3-30B-A3B
[ -d "$DRAFT_MODEL" ] || export DRAFT_MODEL=/workspace/jetspec-v2/models/jetspec-qwen3-30b-a3b
export HF_DATASETS_CACHE=/workspace/jetspec-v2/.hf-datasets-cache

MAX_SAMPLES="${1:-16}"
TAG="$([ "$MAX_SAMPLES" = "0" ] && echo full || echo smoke${MAX_SAMPLES})"
# Include MoE backend in the output dir so different kernels don't resume each other's results.
export PROFILER_DIR=/workspace/jetspec-v2/out/math500-${TAG}-${JETSPEC_MOE_BACKEND}

mkdir -p "$HF_DATASETS_CACHE" "$PROFILER_DIR"
cd "$VLLM_FORK_DIR"

# TP=1 (single GPU; README example used TP=4). triton tree kernel (SM90 optimus won't run on sm_120).
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}" \
bash examples/offline_inference/jetspec_profiling_math500_tree_budget_bsz_sweep.sh \
  --model "${TARGET_MODEL}" \
  --draft-model "${DRAFT_MODEL}" \
  --profiler-dir "${PROFILER_DIR}" \
  --tree-attn-kernel triton \
  --enable-expert-parallel \
  --disable-cascade-attn \
  --cudagraph-mode default \
  --tp-size 1 \
  --batch-sizes 1 \
  --max-num-seqs 1 \
  --tree-budgets "${TREE_BUDGETS:-128}" \
  --max-tokens "${MAX_TOKENS:-512}" \
  --max-samples "${MAX_SAMPLES}" \
  --num-warmup-runs 1 \
  --profiler none \
  --max-model-len 3072 \
  --max-num-batched-tokens 16384
