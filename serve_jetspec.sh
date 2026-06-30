#!/usr/bin/env bash
# JetSpec-enabled vLLM OpenAI-compatible server on :8000 (RTX PRO 6000 Blackwell / sm_120).
# Mirrors the validated offline config: CUDA 12.9 nvcc for FlashInfer (NOT its lib64 on
# LD_LIBRARY_PATH), triton tree-attn, CUTLASS MoE, dflash tree spec at budget 128.
set -euo pipefail

source /workspace/jetspec-v2/.venv/bin/activate

# CUDA 12.9 toolkit for FlashInfer's nvcc/headers only. Do NOT add its lib64 to LD_LIBRARY_PATH
# (its libcublas/cudart would shadow torch's cu128 runtime -> CUBLAS_STATUS_INVALID_VALUE).
export CUDA_HOME=/root/cuda-12.9
export PATH="$CUDA_HOME/bin:$PATH"

PORT="${PORT:-8000}"
MAXLEN="${MAXLEN:-8192}"
# tree_width>1 = JetSpec tree drafting (batch-1 only; breaks at batch>1).
# tree_width=1 = DFlash linear drafting (batch-safe). Default here: linear.
TREE_WIDTH="${TREE_WIDTH:-1}"
if [ "$TREE_WIDTH" -gt 1 ]; then BUDGET="${MAX_TREE_BUDGET:-128}"; VARIANT="jetspec-tree"; else BUDGET="${MAX_TREE_BUDGET:-16}"; VARIANT="dflash-linear"; fi
# linear is batch-safe -> allow real batching by default; tree must stay single-stream.
if [ "$TREE_WIDTH" -gt 1 ]; then MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"; else MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"; fi

# Prefer RAM-staged weights; fall back to disk.
TARGET=/dev/shm/models/Qwen3-30B-A3B
DRAFT=/dev/shm/models/jetspec-qwen3-30b-a3b
[ -d "$TARGET" ] || TARGET=/workspace/jetspec-v2/models/Qwen3-30B-A3B
[ -d "$DRAFT" ]  || DRAFT=/workspace/jetspec-v2/models/jetspec-qwen3-30b-a3b

cd /workspace/jetspec-v2/vllm-jetspec

SPEC_CONFIG="{\"method\":\"dflash\",\"model\":\"${DRAFT}\",\"num_speculative_tokens\":15,\
\"head_type\":\"causal\",\"tree_width\":${TREE_WIDTH},\"max_tree_budget\":${BUDGET},\"tree_draft\":\"accum_logp\",\
\"max_draft_passes\":0,\"tree_prune_ratio\":0.25,\"tree_construction\":\"breadth_first\",\
\"tree_attn_kernel\":\"triton\",\"tree_kv_layout\":\"logical\",\"num_cudagraph_tree_captures\":4,\
\"max_model_len\":${MAXLEN}}"

echo "Serving variant=${VARIANT} tree_width=${TREE_WIDTH} budget=${BUDGET} max_num_seqs=${MAX_NUM_SEQS} on :${PORT}"
exec vllm serve "$TARGET" \
  --served-model-name "${VARIANT}-qwen3-30b-a3b" \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  --moe-backend flashinfer_cutlass \
  --gpu-memory-utilization 0.90 \
  --max-model-len "$MAXLEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --host 0.0.0.0 --port "$PORT" \
  --speculative-config "$SPEC_CONFIG"
