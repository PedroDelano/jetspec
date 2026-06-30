#!/usr/bin/env python3
"""Batch correctness check for JetSpec vs AR on Qwen3-30B-A3B (sm_120).

Runs a batch of DISTINCT prompts (varied type/length) concurrently at temp=0.
JetSpec is greedy-lossless, so dflash output must match AR token-for-token and be
coherent (no gibberish). Run once per mode, then diff the JSONs.

Usage: python batch_check.py {ar|dflash}
"""
import os, sys, json
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

mode = sys.argv[1] if len(sys.argv) > 1 else "dflash"
assert mode in ("ar", "dflash")

def pick(p):
    return p if os.path.isdir(p) else p.replace("/dev/shm/models", "/workspace/jetspec-v2/models")
TARGET = pick("/dev/shm/models/Qwen3-30B-A3B")
DRAFT  = pick("/dev/shm/models/jetspec-qwen3-30b-a3b")

# 8 DISTINCT prompts — varied domain and length (heterogeneous batch).
prompts_raw = [
    "What is 17 * 23? Show your reasoning step by step.",
    "Write a Python function `is_palindrome(s)` that ignores case and spaces.",
    "In two sentences, explain why the sky appears blue.",
    "Solve for x: 3x - 7 = 2x + 5.",
    "Write a haiku about the ocean at night.",
    "List the first 6 prime numbers, separated by commas.",
    "Translate 'good morning, my friend' into Spanish and German.",
    "A train travels 60 km in 45 minutes. What is its average speed in km/h?",
]

tok = AutoTokenizer.from_pretrained(TARGET, trust_remote_code=True)
prompts = [
    tok.apply_chat_template([{"role": "user", "content": p}],
                            tokenize=False, add_generation_prompt=True)
    for p in prompts_raw
]

kw = dict(
    model=TARGET, trust_remote_code=True, tensor_parallel_size=1,
    enable_expert_parallel=True, moe_backend="flashinfer_cutlass",
    gpu_memory_utilization=0.90, max_model_len=4096, max_num_seqs=len(prompts),
)
if mode == "dflash":
    kw["speculative_config"] = {
        "method": "dflash", "model": DRAFT, "num_speculative_tokens": 15,
        "head_type": "causal", "tree_width": 7, "max_tree_budget": 128,
        "tree_draft": "accum_logp", "max_draft_passes": 0, "tree_prune_ratio": 0.25,
        "tree_construction": "breadth_first", "tree_attn_kernel": "triton",
        "tree_kv_layout": "logical", "num_cudagraph_tree_captures": 4,
        "max_model_len": 4096,
    }

llm = LLM(**kw)
sp = SamplingParams(temperature=0.0, max_tokens=256)
outs = llm.generate(prompts, sp)   # all prompts submitted together => batched

recs = []
for i, o in enumerate(outs):
    t = o.outputs[0].text
    recs.append({"i": i, "prompt": prompts_raw[i], "text": t,
                 "n_tokens": len(o.outputs[0].token_ids),
                 "token_ids": list(o.outputs[0].token_ids)})
out_path = f"/workspace/jetspec-v2/batchcheck_{mode}.json"
json.dump(recs, open(out_path, "w"), indent=2)
print(f"\n===== {mode.upper()} batch outputs ({len(recs)} distinct prompts) =====")
for r in recs:
    print(f"\n--- [{r['i']}] {r['prompt']}  ({r['n_tokens']} tok)")
    print(r["text"][:500])
print(f"\nWrote {out_path}")
