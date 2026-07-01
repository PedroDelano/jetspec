# JetSpec via vLLM on RTX PRO 6000 Blackwell (sm_120)

Reproduction of [**JetSpec**](https://github.com/hao-ai-lab/JetSpec) (Hao AI Lab — "causal parallel
tree drafting" speculative decoding) through its vLLM v1 fork
[`JetSpec-project/vllm-jetspec`](https://github.com/JetSpec-project/vllm-jetspec), on a single
**NVIDIA RTX PRO 6000 Blackwell Server Edition (sm_120, 96 GB)**.

Target model: **`Qwen/Qwen3-30B-A3B`** (MoE, 30B total / 3B active) + draft head
**`JetSpec/jetspec-qwen3-30b-a3b`**.

> **TL;DR**
> - The full JetSpec/vLLM stack **runs end-to-end on Blackwell sm_120** (nontrivial — see Setup).
> - JetSpec's tree drafting **reproduces faithfully**: MATH-500 acceptance length **8.7–9.7** (paper 9.56), lossless.
> - **Single-stream speedup ≈ 1.83×** (budget 128) — not the paper's 9.64×, capped by sm_120 kernel maturity (no SM90 tree-attn kernel; CUTLASS MoE on sm_120 uses sm80-fallback tactics; MoE verify heavier than the paper's dense 8B).
> - ⚠️ **JetSpec tree drafting is broken at batch > 1**: GSM8K accuracy collapses 85% → ~10–16% with gibberish. The **linear DFlash** drafter and the **AR** baseline stay correct in batch. JetSpec tree is only safe single-stream.
> - ✅ **Reproducible from scratch** (verified): `bash setup_fresh.sh <dir>` in a brand-new folder — `git clone` → build from the lock → verify `.so` → smoke-test — passed clean (JetSpec generated `42` for 6×7). Build on **local disk**, not the `/workspace` NFS mount.

## Results

### Single-stream speedup (MATH-500, batch 1, tuned harness, FlashInfer CUTLASS MoE)
| MoE backend | tree budget | AR tok/s | JetSpec tok/s | speedup | accept-len |
|---|---|---|---|---|---|
| Triton (sm_120 fallback) | 128 | 118 | 89 | **0.76×** | 10.7 |
| FlashInfer CUTLASS | 32 | 146 | 196 | 1.34× | 5.0 |
| FlashInfer CUTLASS | 64 | 146 | 242 | 1.65× | 7.1 |
| **FlashInfer CUTLASS** | **128** | 146 | **267** | **1.83×** | 8.7 |
| FlashInfer CUTLASS | 256 | 146 | 253 | 1.73× | 9.2 |
| FlashInfer CUTLASS | 512 | 146 | 219 | 1.49× | 9.7 |

Peak at budget 128. The verify step (≤128 tree tokens) costs ~4.8× a single decode here vs ~1× on
B200 with the SM90 "optimus" kernel — that ratio is the whole speedup gap.

### Batch correctness (GSM8K, N=80, multiprocessing on)
| method | batch 1 | batch 4 | batch 8 | batch 16 |
|---|---|---|---|---|
| **JetSpec-tree** (tree_width 7) | 85% | **10%** | **9%** | **16%** |
| DFlash-linear (tree_width 1) | 85% | — | 84% | 83% |
| AR baseline | 84% | — | 79% | 85% |

JetSpec tree drafting corrupts output once `max_num_seqs > 1` (gibberish, e.g. `0707777…`). AR and
the linear drafter are unaffected → the bug is **specific to the batched tree-spec path**, not vLLM
batching/multiprocessing/the model.

### Three-way comparison (GSM8K, accuracy + throughput)
| config | batch | accuracy | gibberish | tok/s | speedup† |
|---|---|---|---|---|---|
| AR baseline | 1 | 83.8% | 0 | 153.8 | 1.00× |
| DFlash-linear | 1 | 85.0% | 0 | 159.6 | 1.04× |
| JetSpec-tree | 1 | 82.5% | 1 | 150.6 | 0.98× |
| AR baseline | 8 | 78.8% | 0 | 431.4 | 1.00× |
| **DFlash-linear** | 8 | **83.8%** ✅ | 0 | 651.5 | **1.51×** |
| JetSpec-tree | 8 | **13.8%** ❌ | 13 | 750.4 | 1.74× (garbage) |

†vs AR at the same batch. **Note:** this simple `LLM.generate` harness does not graph the draft head,
so its **batch-1 tok/s understate spec-decode speed** — the authoritative single-stream number is the
tuned MATH-500 result (1.83×). Accuracy numbers are valid.

**Bottom line:** DFlash-linear is the batch-safe accelerator (correct + speedup that grows with batch);
JetSpec-tree is faster single-stream but unusable beyond batch 1 until the batched tree path is fixed.

## Hardware
1× RTX PRO 6000 Blackwell Server Edition · sm_120 · 96 GB GDDR7 (~1792 GB/s) · driver 580 (CUDA 13
capable) · system nvcc 12.1 · 2× EPYC 9555 · 1.5 TB RAM · Ubuntu 22.04. See `HARDWARE.md`.

## Setup gotchas (why README-from-scratch fails on sm_120)
1. **torch 2.10.0+cu128** — not 2.9 (vLLM 0.18-dev needs `torch._dynamo.convert_frame.GraphCaptureOutput`).
2. **Precompiled vLLM `.so`** from `wheels.vllm.ai/<base-commit 551b3fb>/cu129/` (avoids building CUDA on nvcc 12.1). The NFS filesystem truncates extraction — verify `vllm/_C.abi3.so` and `_moe_C.abi3.so` exist; re-extract from the wheel if missing.
3. **transformers `>=4.56,<5`** (an unconstrained reinstall grabs 5.x and breaks).
4. **CUDA 12.9 toolkit** at `/root/cuda-12.9` so FlashInfer can JIT the CUTLASS MoE for sm_120 — but **do NOT put `$CUDA_HOME/lib64` on `LD_LIBRARY_PATH`** (its libcublas/cudart shadow torch's cu128 runtime → `CUBLAS_STATUS_INVALID_VALUE`).
5. **MoE backend = `flashinfer_cutlass`** (CUDA 12.9). The auto-pick / Triton MoE is ~2.4× slower (0.76× vs 1.83×). FlashInfer TRTLLM MoE is sm100-only.
6. **Tree-attn kernel = `triton`** — the SM90 "optimus" CuTe kernel can't run on sm_120.
7. **Stage weights in `/dev/shm`** (tmpfs) — model load 846 s → ~25 s.

Fork edits are captured in `patches/vllm-jetspec.local.patch` (set `moe_backend`; drop the toolkit-lib64 prepend).

### Reproducible environment (pinned)
Exact versions are locked in **`requirements.lock.txt`** (`uv pip freeze`, 189 pkgs) and **`STACK_LOCK.md`**
(toolchain, CUDA 12.9.1, vLLM base commit `551b3fb`, model HF revisions). Rebuild from the lock, not the
ad-hoc `cycle/` commands:
**Turnkey:** `bash setup_fresh.sh <fresh-dir> [models-src]` clones the repo + `vllm-jetspec`, builds the
venv from the lock, verifies/repairs the precompiled `.so`, links weights, and smoke-tests. **Build on
local disk (e.g. `/root`), not the `/workspace` NFS mount** (NFS intermittently corrupts pip extraction).
Manual equivalent:
```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install "setuptools>=77,<81" setuptools-scm wheel packaging cmake ninja jinja2 regex build
VLLM_USE_PRECOMPILED=1 uv pip install -e vllm-jetspec --no-build-isolation
uv pip install --no-deps -r requirements.lock.txt   # --no-deps: install the full freeze verbatim
```

## How to run
```bash
# Throughput / acceptance (MATH-500, tuned harness):
JETSPEC_MOE_BACKEND=flashinfer_cutlass TREE_BUDGETS=128 bash run_math500.sh 32

# GSM8K accuracy at a batch size:  python gsm8k_acc.py {ar|dflash} <max_num_seqs> <n>   (TREE_WIDTH env)
bash run_gsm8k_sweep.sh        # JetSpec-tree across batch sizes + AR reference
bash run_compare.sh            # AR vs DFlash-linear vs JetSpec-tree (acc + tok/s) at batch 1 & 8

# OpenAI-compatible server on :8000 (default DFlash-linear, batch-safe):
bash serve_jetspec.sh                  # tree_width=1, max_num_seqs=8
TREE_WIDTH=7 bash serve_jetspec.sh     # JetSpec-tree (single-stream only)
curl localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"dflash-linear-qwen3-30b-a3b","prompt":"What is 17*23? Answer:","max_tokens":64,"temperature":0}'
```

## Repo layout
| path | what |
|---|---|
| `GOAL.md` `RESEARCH.md` `HARDWARE.md` `RULES.md` | reproduction goal, research, hardware, iron rules |
| `FINAL.md` | runbook + original-vs-ours verdict |
| `cycle/001‑007` | per-cycle log: install → torch fix → precompiled `.so` → fast MoE → budget sweep → batch correctness → 3-way comparison |
| `run_math500.sh` `run_gsm8k_sweep.sh` `run_linear_contrast.sh` `run_compare.sh` | run scripts |
| `gsm8k_acc.py` `batch_check.py` | accuracy / batch-correctness harnesses |
| `serve_jetspec.sh` | OpenAI-compatible server (DFlash-linear default) |
| `gsm8k_results.csv` `gsm8k_compare.csv` | measured results |
| `patches/vllm-jetspec.local.patch` | the two fork edits |

The model weights, `.venv`, the `vllm-jetspec` clone, and `out/` are gitignored.
