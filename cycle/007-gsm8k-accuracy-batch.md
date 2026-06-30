# Cycle 007 — GSM8K accuracy vs batch size

## Goal (user)
Run GSM8K at different batch sizes to see if accuracy drops — quantitative test of the
batch/multiprocessing gibberish concern. JetSpec is lossless → accuracy should be ~flat across
batch sizes and ~equal to AR. A drop at high batch = real batch bug.

## Context from cycle 006
JetSpec batch=8 (multiprocessing ON) produced coherent output for 7/8 distinct prompts, but
prompt [6] (translation) degenerated to repeated digits (`0707777…`). GSM8K acc-vs-batch will
quantify whether this kind of degeneration hurts correctness and scales with batch size.

## Method
`gsm8k_acc.py {ar|dflash} <max_num_seqs> <n>`: openai/gsm8k main test, N=80, chat template asking
for `\boxed{N}`, temp=0, max_tokens=1024, CUTLASS MoE, triton tree-attn, budget 128, /dev/shm.
`run_gsm8k_sweep.sh` sweeps dflash max_num_seqs ∈ {1,4,8,16} + AR@16 reference, VLLM_ENABLE_V1_MULTIPROCESSING=1.
Answer extraction: last \boxed{} else last number. Also flags outputs with a char repeated >30x (gibberish).

## Measured (N=80, GSM8K, multiprocessing ON)
| mode | max_num_seqs | accuracy | gibberish |
|---|---|---|---|
| dflash | 1 | **0.850** (68/80) | 0 |
| dflash | 4 | **0.100** (8/80) | 5 |
| dflash (tree, tw7) | 8 | **0.0875** (7/80) | 15 |
| dflash (tree, tw7) | 16 | **0.1625** (13/80) | 7 |
| **ar** | **16** | **0.8500** (68/80) | **0** ✅ |
| **dflash linear (tw1)** | **8** | **0.8125** (65/80) | **0** ✅ |
| **dflash linear (tw1)** | **16** | **0.8250** (66/80) | **0** ✅ |

## Conclusion
**JetSpec's parallel TREE drafting (tree_width>1) is broken at batch>1.** Accuracy collapses
85% (batch1) → ~10% (batch≥4) with gibberish, while every control stays healthy in batch:
AR@16 = 85%/0, linear DFlash (tw1) @8 = 81%/0, @16 = 83%/0. So the corruption is **specific to
the batched tree-spec path** (tree drafting / tree attention / tree verify) — NOT vLLM batching,
NOT multiprocessing, NOT the model, NOT the linear drafter. Same failure class the user saw in
DFlash. JetSpec tree mode is only safe at batch=1 (single-stream) on this setup.

Artifacts: gsm8k_results.csv, gsm8k_{dflash,dflash_tw1,ar}_mns*.json (per-problem gold/pred/gibberish).

**AR@16 = 85% / 0 gibberish — identical to JetSpec@batch1.** → The model, vLLM batching, and
multiprocessing are correct in batch. The accuracy collapse is **JetSpec-tree-specific**, not a
general batching issue.

**Finding: JetSpec tree drafting accuracy COLLAPSES at batch>1** — 85% (batch1) → 10/9/16% (batch 4/8/16),
with gibberish outputs appearing only at batch>1. Confirmed it IS JetSpec (method=dflash + tree_width=7;
no separate "jetspec" method exists in the fork — tree_width>1 engages JetSpec's parallel tree drafting).
Pending: AR@16 (is the model/batching fine? expect ~85%) and linear DFlash tw1@{8,16} (is the bug
tree-specific or general dflash batching?).
