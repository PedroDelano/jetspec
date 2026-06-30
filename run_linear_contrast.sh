#!/usr/bin/env bash
# Linear DFlash (tree_width=1) GSM8K at batch 8 & 16, to contrast with JetSpec tree drafting.
# If linear ALSO collapses in batch -> bug is general dflash batching; if linear holds -> tree-specific.
set -uo pipefail
source /workspace/jetspec-v2/.venv/bin/activate
export CUDA_HOME=/root/cuda-12.9
export PATH="$CUDA_HOME/bin:$PATH"
export VLLM_ENABLE_V1_MULTIPROCESSING=1
N="${N:-80}"

# Wait for the main tree sweep to finish (it prints this marker at the end).
until grep -q "=== gsm8k_results.csv ===" /workspace/jetspec-v2/.gsm8k_sweep.log 2>/dev/null; do sleep 20; done
echo ">>> main sweep done; starting linear contrast $(date)"

for MNS in 8 16; do
  echo ">>> LINEAR DFlash tree_width=1 GSM8K max_num_seqs=$MNS n=$N $(date)"
  TREE_WIDTH=1 python /workspace/jetspec-v2/gsm8k_acc.py dflash "$MNS" "$N" 2>&1 \
    | grep -E "RESULT|Traceback|Error:|RuntimeError|gibberish" | tail -4
done
echo "=== FINAL gsm8k_results.csv ==="
cat /workspace/jetspec-v2/gsm8k_results.csv
