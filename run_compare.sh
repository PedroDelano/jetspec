#!/usr/bin/env bash
# Three-way comparison on GSM8K: AR baseline vs DFlash-linear (tw1) vs JetSpec-tree (tw7).
# Captures accuracy + tok/s at batch 1 and batch 8. Fresh process per cell (clean GPU).
set -uo pipefail
source /workspace/jetspec-v2/.venv/bin/activate
export CUDA_HOME=/root/cuda-12.9
export PATH="$CUDA_HOME/bin:$PATH"
export VLLM_ENABLE_V1_MULTIPROCESSING=1
N="${N:-80}"

echo "mode,max_num_seqs,n,accuracy,correct,gibberish,tok_s,out_tokens,elapsed_s" > /workspace/jetspec-v2/gsm8k_compare.csv

run() { # mode tree_width mns
  echo ">>> $1 (tw=$2) mns=$3 n=$N  $(date)"
  TREE_WIDTH="$2" python /workspace/jetspec-v2/gsm8k_acc.py "$1" "$3" "$N" 2>&1 \
    | grep -E "RESULT|Traceback|RuntimeError:|Error:" | tail -3
}

for MNS in 1 8; do
  run ar     7 "$MNS"      # baseline (tree_width ignored for ar)
  run dflash 1 "$MNS"      # DFlash linear
  run dflash 7 "$MNS"      # JetSpec tree
done

echo "=== gsm8k_compare.csv ==="
cat /workspace/jetspec-v2/gsm8k_compare.csv
