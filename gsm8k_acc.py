#!/usr/bin/env python3
"""GSM8K accuracy for JetSpec vs AR at a given batch size (max_num_seqs).

Goal: detect whether accuracy degrades as batch size grows (the batch/multiprocessing
gibberish concern). JetSpec is lossless, so accuracy should be ~constant across batch sizes
and ~equal to AR. A drop at high batch = a real batch bug.

Usage: python gsm8k_acc.py {ar|dflash} <max_num_seqs> <n_samples>
Appends one row to gsm8k_results.csv.
"""
import os, sys, re, json
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

mode = sys.argv[1]; MNS = int(sys.argv[2]); N = int(sys.argv[3])
assert mode in ("ar", "dflash")
TREE_WIDTH = int(os.environ.get("TREE_WIDTH", "7"))   # >1 = JetSpec tree drafting; 1 = linear DFlash

def pick(p): return p if os.path.isdir(p) else p.replace("/dev/shm/models", "/workspace/jetspec-v2/models")
TARGET = pick("/dev/shm/models/Qwen3-30B-A3B")
DRAFT  = pick("/dev/shm/models/jetspec-qwen3-30b-a3b")

ds = load_dataset("openai/gsm8k", "main", split="test").select(range(N))
def gold_of(ans): return ans.split("####")[-1].strip().replace(",", "")
golds = [gold_of(a) for a in ds["answer"]]

tok = AutoTokenizer.from_pretrained(TARGET, trust_remote_code=True)
SYS = "Solve the math problem. End your response with 'The final answer is \\boxed{N}' where N is the number."
prompts = [tok.apply_chat_template(
    [{"role": "user", "content": SYS + "\n\n" + q}],
    tokenize=False, add_generation_prompt=True) for q in ds["question"]]

kw = dict(model=TARGET, trust_remote_code=True, tensor_parallel_size=1,
          enable_expert_parallel=True, moe_backend="flashinfer_cutlass",
          gpu_memory_utilization=0.90, max_model_len=4096, max_num_seqs=MNS)
if mode == "dflash":
    kw["speculative_config"] = {"method": "dflash", "model": DRAFT, "num_speculative_tokens": 15,
        "head_type": "causal", "tree_width": TREE_WIDTH, "max_tree_budget": (128 if TREE_WIDTH > 1 else 16), "tree_draft": "accum_logp",
        "max_draft_passes": 0, "tree_prune_ratio": 0.25, "tree_construction": "breadth_first",
        "tree_attn_kernel": "triton", "tree_kv_layout": "logical", "num_cudagraph_tree_captures": 4,
        "max_model_len": 4096}

llm = LLM(**kw)
outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=1024))

def pred_of(text):
    m = re.findall(r"\\boxed\{([^}]*)\}", text)
    s = m[-1] if m else None
    if s is None:
        nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
        s = nums[-1] if nums else ""
    s = s.replace(",", "").replace("$", "").strip().rstrip(".")
    try: return str(int(float(s)))
    except Exception: return s

correct = 0; gibberish = 0; preds = []
for o, g in zip(outs, golds):
    t = o.outputs[0].text
    p = pred_of(t)
    ok = (p == g) or (p.replace(".0","") == g)
    correct += ok
    # crude gibberish flag: a single char repeated >30x in a row
    if re.search(r"(.)\1{30,}", t): gibberish += 1
    preds.append({"gold": g, "pred": p, "ok": bool(ok), "ntok": len(o.outputs[0].token_ids)})

acc = correct / len(golds)
label = mode if mode == "ar" else f"dflash_tw{TREE_WIDTH}"
row = f"{label},{MNS},{N},{acc:.4f},{correct},{gibberish}"
print(f"\n===== RESULT mode={label} max_num_seqs={MNS} n={N}: accuracy={acc:.4f} ({correct}/{N}), gibberish_outputs={gibberish}")
with open("/workspace/jetspec-v2/gsm8k_results.csv", "a") as f:
    f.write(row + "\n")
json.dump(preds, open(f"/workspace/jetspec-v2/gsm8k_{label}_mns{MNS}.json", "w"), indent=2)
