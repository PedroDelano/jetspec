#!/usr/bin/env bash
# GSM8K accuracy sweep across batch sizes (max_num_seqs) for JetSpec, + AR reference.
# Detects whether accuracy drops as batch size grows. Fresh process per config = clean GPU.
set -uo pipefail
source /workspace/jetspec-v2/.venv/bin/activate
export CUDA_HOME=/root/cuda-12.9
export PATH="$CUDA_HOME/bin:$PATH"
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-1}"   # multiprocessing path

N="${N:-80}"
echo "mode,max_num_seqs,n,accuracy,correct,gibberish_outputs" > /workspace/jetspec-v2/gsm8k_results.csv

for MNS in 1 4 8 16; do
  echo ">>> JetSpec GSM8K  max_num_seqs=$MNS  n=$N  $(date)"
  python /workspace/jetspec-v2/gsm8k_acc.py dflash "$MNS" "$N" 2>&1 \
    | grep -E "RESULT|Traceback|Error:|RuntimeError|gibberish" | tail -5
done
# AR reference at the largest batch (lossless target: dflash should match this accuracy)
echo ">>> AR GSM8K  max_num_seqs=16  n=$N  $(date)"
python /workspace/jetspec-v2/gsm8k_acc.py ar 16 "$N" 2>&1 \
  | grep -E "RESULT|Traceback|Error:|RuntimeError|gibberish" | tail -5

echo "=== gsm8k_results.csv ==="
cat /workspace/jetspec-v2/gsm8k_results.csv
