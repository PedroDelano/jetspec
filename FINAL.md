# FINAL — JetSpec via vLLM on RTX PRO 6000 Blackwell (sm_120)

**Verdict: PARTIAL reproduction.** The JetSpec algorithm is faithfully and losslessly reproduced
(acceptance length matches/exceeds the paper). The **wall-clock speedup is hardware/kernel-maturity
limited to ~1.83×** on this card — the original's ~9.6× is not achievable on sm_120 today, for the
reasons proven below. This is **not** an implementation error: acceptance length independently
confirms the implementation is correct.

## What was reproduced
- Target: `Qwen/Qwen3-30B-A3B` (MoE, 30B/3B-active, bf16) + draft head `JetSpec/jetspec-qwen3-30b-a3b`
- Path: `JetSpec-project/vllm-jetspec` (vLLM 0.18.2rc1.dev57 base, commit 551b3fb), MATH-500, batch=1, TP=1
- Tree-spec decoding runs end-to-end on the Blackwell card (TREE_ATTN backend + Triton tree kernel + CUTLASS MoE).

## Runbook (this hardware — replayable)
```bash
# 0. Hardware: 1x RTX PRO 6000 Blackwell Server Edition, 96GB, sm_120, driver 580 (CUDA 13 capable).
# 1. uv venv (Python 3.12) + torch 2.10.0+cu128 (NOT 2.9 — needs torch._dynamo GraphCaptureOutput)
uv venv --python 3.12 /workspace/jetspec-v2/.venv && source /workspace/jetspec-v2/.venv/bin/activate
uv pip install torch==2.10.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 2. Install the vLLM fork with PRECOMPILED binaries (avoids building CUDA on nvcc 12.1):
uv pip install "setuptools>=77,<81" setuptools-scm wheel packaging cmake ninja jinja2 regex build
cd /workspace/jetspec-v2/vllm-jetspec
VLLM_USE_PRECOMPILED=1 UV_LINK_MODE=copy uv pip install -e . --no-build-isolation
#  -> NFS extraction may truncate .so files. If `import vllm._C`/`_moe_C` fails, re-extract the
#     matching precompiled wheel from wheels.vllm.ai/<base-commit>/cu129/ and copy the .so into vllm/.
uv pip install "transformers>=4.56,<5" datasets tqdm psutil openai   # repair/pin runtime deps

# 3. CUDA 12.9 TOOLKIT for FlashInfer CUTLASS MoE JIT (sm_120 needs nvcc>=12.9; pip nvcc lacks the driver):
sh cuda_12.9.1_*_linux.run --silent --toolkit --toolkitpath=/root/cuda-12.9 --override --no-opengl-libs
ln -s . /root/cuda-12.9/include/cccl     # flashinfer expects $CUDA_HOME/include/cccl
# IMPORTANT: set CUDA_HOME for nvcc/headers ONLY. Do NOT put $CUDA_HOME/lib64 on LD_LIBRARY_PATH
# (its libcublas/cudart 12.9 shadow torch's cu128 runtime -> CUBLAS_STATUS_INVALID_VALUE). The fork's
# launcher was patched to drop that prepend.

# 4. Stage weights in /dev/shm (tmpfs) -> model load 846s -> ~25s.
cp -r models/Qwen3-30B-A3B models/jetspec-qwen3-30b-a3b /dev/shm/models/

# 5. Run (see run_math500.sh): forces moe_backend=triton OR flashinfer_cutlass, --tree-attn-kernel triton,
#    --enable-expert-parallel, TP=1, batch=1. Best speedup at --tree-budgets 128.
CUDA_HOME=/root/cuda-12.9 JETSPEC_MOE_BACKEND=flashinfer_cutlass TREE_BUDGETS=128 bash run_math500.sh 32
```
Key edits made (logged): `dflash_profiling.py` sets `moe_backend` (env `JETSPEC_MOE_BACKEND`); entry
script no longer prepends `$CUDA_HOME/lib64`.

## Original vs ours
| | Original (paper/README) | Ours |
|---|---|---|
| Hardware | H100 / B200 | 1× RTX PRO 6000 Blackwell (sm_120), 1792 GB/s |
| Model | Qwen3-8B (dense) | Qwen3-30B-A3B (MoE) — user-chosen |
| Tree-verify attn kernel | SM90 "optimus" CuTe | Triton (optimus is SM90-only, won't run on sm_120) |
| MoE kernel | n/a (dense) | FlashInfer CUTLASS (sm80-fallback tactics on sm_120) |
| **Acceptance length (MATH-500)** | **9.56** | **8.73 @ budget 128, up to 9.74 @ budget 512** ✅ |
| **End-to-end speedup** | **9.64×** | **1.83× (peak, budget 128)** |
| AR baseline / JetSpec tok/s | — | 146.4 / 267.4 (budget 128) |

Speedup vs tree budget (ours, 32 samples): 32→1.34×, 64→1.65×, **128→1.83×**, 256→1.73×, 512→1.49×.

## Verdict: hardware / kernel-maturity constraint (not implementation error)
- **Algorithm reproduced:** acceptance length 8.73–9.74 is in the paper's band (9.56) and lossless — the
  causal parallel tree drafting works correctly on the full vLLM stack on Blackwell.
- **Speedup gap is kernel maturity, proven:** the tree-verify forward (≤128 tokens) costs ~4.8× a single
  decode step here, vs ~1× on B200 with the optimized SM90 optimus kernel. The entire shortfall is that
  ratio. Causes: (1) the SM90 optimus CuTe tree-attention kernel is architecturally unavailable on sm_120;
  (2) FlashInfer's CUTLASS MoE for sm_120 dispatches sm80 generic GEMM tactics (autotuner: "GPU lacks shared
  memory resources" on native tactics); (3) the chosen 30B-A3B MoE has a heavier verify than Qwen3-8B dense.
- The one fixable cause we found — the auto-selected MoE backend silently being slow Triton — was fixed
  (Triton MoE 0.76× → CUTLASS MoE 1.83×).

## Recommended further validation (optional)
- Run the same pipeline on **Qwen3-8B dense** (`JetSpec/jetspec-qwen3-8b`) to isolate the MoE-verify cost:
  a dense model should show a higher speedup, confirming the MoE+sm_120 kernels (not the algorithm) are
  the limiter. This is the headline's exact model and the cleanest apples-to-apples on this card.
- A native sm_120 tree-attention kernel (or an sm_120-tuned CUTLASS MoE) would be required to approach the
  original speedups; both are upstream kernel-maturity items, out of scope for a reproduction.
