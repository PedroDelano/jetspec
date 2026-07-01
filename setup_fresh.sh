#!/usr/bin/env bash
# Clean-room reproduction: build the JetSpec/vLLM stack in a BRAND-NEW folder from the pinned lock,
# then smoke-run vLLM (JetSpec dflash tree) to prove it works. Models are symlinked from an existing
# pinned download (re-downloading 57GB proves nothing new); pass MODELS_SRC to override.
set -euo pipefail

ROOT="${1:?usage: setup_fresh.sh <fresh-dir> [models-src]}"
MODELS_SRC="${2:-/workspace/jetspec-v2/models}"
REPO_URL="git@github.com:PedroDelano/jetspec.git"
FORK_URL="https://github.com/JetSpec-project/vllm-jetspec.git"
FORK_COMMIT="f90d5ca17a2c05f436a80ee2e0984cc7a22e1a16"
VLLM_BASE_COMMIT="551b3fb39f3a95ff3dc3feca9528ab4c90649316"
WHEEL="https://wheels.vllm.ai/${VLLM_BASE_COMMIT}/vllm-0.18.2rc1.dev57%2Bg551b3fb39-cp38-abi3-manylinux_2_31_x86_64.whl"
export UV_LINK_MODE=copy CUDA_HOME=/root/cuda-12.9 PATH="/root/cuda-12.9/bin:$PATH"

# The /workspace NFS intermittently throws "Stale file handle" mid-extraction; retry installs.
retry() { local n=0; until "$@"; do n=$((n+1)); [ "$n" -ge 5 ] && { echo "FAILED after $n tries: $*"; return 1; }; echo "  retry $n/4: $*"; sleep 5; done; }
export -f retry

echo "===[1/7] fresh folder: $ROOT"
rm -rf "$ROOT"; mkdir -p "$ROOT"; cd "$ROOT"

echo "===[2/7] git clone deliverables from GitHub (proves the push is self-sufficient)"
git clone "$REPO_URL" repo

echo "===[3/7] clone vllm-jetspec @ pinned commit + apply clean patch"
git clone "$FORK_URL" vllm-jetspec
( cd vllm-jetspec
  [ "$(git rev-parse HEAD)" = "$FORK_COMMIT" ] || { git fetch --depth 1 origin "$FORK_COMMIT" && git checkout "$FORK_COMMIT"; }
  git apply ../repo/patches/vllm-jetspec.local.patch && echo "patch applied" )

echo "===[4/7] uv venv + pinned torch (cu128) + precompiled vLLM + locked deps"
uv venv --python 3.12 .venv
source .venv/bin/activate
retry uv pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
# build deps must exist BEFORE the editable build (--no-build-isolation runs setup.py against this env)
retry uv pip install "setuptools>=77.0.3,<81.0.0" "setuptools-scm>=8" wheel "packaging>=24.2" "cmake>=3.26.1" ninja "jinja2>=3.1.6" regex build
( cd vllm-jetspec && VLLM_USE_PRECOMPILED=1 retry uv pip install -e . --no-build-isolation )
# install the rest of the locked set, minus the absolute-path editable vllm line (installed above).
# --no-deps: it's a full freeze, so install EXACT pinned versions without re-resolving (the freeze is
# internally inconsistent by uv's resolver — cuda-python 13.3.1 wants cuda-bindings>=13.3.1 but the
# working env pins 12.9.4; harmless at runtime, so restore versions verbatim).
grep -v 'vllm-jetspec' repo/requirements.lock.txt > /tmp/req.fresh.txt
retry uv pip install --no-deps -r /tmp/req.fresh.txt

echo "===[5/7] verify/repair precompiled .so (NFS extraction can truncate them)"
python - "$WHEEL" <<'PY'
import os, sys, zipfile, urllib.request, shutil
need = ["vllm/_C.abi3.so","vllm/_C_stable_libtorch.abi3.so","vllm/_moe_C.abi3.so",
        "vllm/_flashmla_C.abi3.so","vllm/_flashmla_extension_C.abi3.so","vllm/cumem_allocator.abi3.so",
        "vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so","vllm/vllm_flash_attn/_vllm_fa3_C.abi3.so"]
tree = "vllm-jetspec"
missing = [f for f in need if not os.path.exists(os.path.join(tree,f))]
sizes_ok = all(os.path.getsize(os.path.join(tree,f))>0 for f in need if os.path.exists(os.path.join(tree,f)))
if not missing and sizes_ok:
    print("all .so present:", len(need)); sys.exit(0)
print("missing/empty:", missing or "size0", "-> re-extracting from wheel")
whl = "/tmp/vllm_pc_fresh.whl"
urllib.request.urlretrieve(sys.argv[1], whl)
z = zipfile.ZipFile(whl); names = set(z.namelist())
for f in need:
    if f not in names: print("  (not in wheel, skip):", f); continue
    info = z.getinfo(f); tgt = os.path.join(tree, f); os.makedirs(os.path.dirname(tgt), exist_ok=True)
    for _ in range(3):
        with z.open(f) as s, open(tgt,"wb") as d: shutil.copyfileobj(s,d,1<<23)
        if os.path.getsize(tgt)==info.file_size: print("  OK", f, info.file_size//1024//1024,"MB"); break
    else: print("  FAIL", f); sys.exit(1)
print("re-extract done")
PY

echo "===[6/7] link pinned weights ($MODELS_SRC)"
mkdir -p models
ln -sfn "$MODELS_SRC/Qwen3-30B-A3B" models/Qwen3-30B-A3B
ln -sfn "$MODELS_SRC/jetspec-qwen3-30b-a3b" models/jetspec-qwen3-30b-a3b

echo "===[7/7] SMOKE: run vLLM JetSpec (dflash tree) on 1 prompt"
python - <<PY
from vllm import LLM, SamplingParams
T="$ROOT/models/Qwen3-30B-A3B"; D="$ROOT/models/jetspec-qwen3-30b-a3b"
llm = LLM(model=T, trust_remote_code=True, tensor_parallel_size=1, enable_expert_parallel=True,
          moe_backend="flashinfer_cutlass", gpu_memory_utilization=0.90, max_model_len=2048, max_num_seqs=1,
          speculative_config={"method":"dflash","model":D,"num_speculative_tokens":15,"head_type":"causal",
            "tree_width":7,"max_tree_budget":128,"tree_draft":"accum_logp","max_draft_passes":0,
            "tree_prune_ratio":0.25,"tree_construction":"breadth_first","tree_attn_kernel":"triton",
            "tree_kv_layout":"logical","num_cudagraph_tree_captures":4,"max_model_len":2048})
o = llm.generate(["Q: What is 6 times 7? Give only the number.\nA:"], SamplingParams(temperature=0.0, max_tokens=24))
print("SMOKE_OUTPUT:", repr(o[0].outputs[0].text.strip()[:120]))
print("FRESH_REPRO_OK")
PY
