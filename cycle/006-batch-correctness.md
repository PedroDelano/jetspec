# Cycle 006 — JetSpec batch correctness (gibberish check)

## Goal
User concern: DFlash produced gibberish in batch/multiprocessing. Verify JetSpec produces correct,
coherent output for a batch of **DISTINCT** prompts. JetSpec is greedy-lossless → at temp=0, dflash
output must match the AR baseline **token-for-token** and be coherent.

## Method
`batch_check.py` submits 8 distinct, varied prompts (math/code/chat, different lengths) together
(true batch, max_num_seqs=8), temp=0, max_tokens=256. Run for `dflash` and `ar`, save token_ids +
text, then diff. Run with **VLLM_ENABLE_V1_MULTIPROCESSING=1** to exercise the multiprocessing path
where DFlash gibberished. CUTLASS MoE, triton tree-attn, budget 128, model from /dev/shm.

## Checks
1. Coherence: are dflash outputs sensible (not repetition/garbage)?
2. Losslessness: do dflash token_ids == ar token_ids per prompt?
3. Cross-contamination: does each batch slot answer ITS OWN prompt (not bleed across requests)?

## Measured
(pending)
