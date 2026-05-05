#!/usr/bin/env bash
# run_finetune_on_runpod.sh
# Uploads finetune_qlora.py to a running RunPod pod, runs QLoRA training,
# and downloads the resulting LoRA adapter.
#
# Usage:
#   bash run_finetune_on_runpod.sh <POD_ID> <SSH_IP> <SSH_PORT>
#
# After success, run:
#   bash upload_adapter_to_kaggle.sh

set -e

POD_ID="${1:?Usage: bash run_finetune_on_runpod.sh POD_ID SSH_IP SSH_PORT}"
SSH_IP="${2:?}"
SSH_PORT="${3:?}"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=30"
SSH="ssh $SSH_OPTS root@${SSH_IP} -p ${SSH_PORT}"
SCP="scp $SSH_OPTS -P ${SSH_PORT}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FINETUNE_PY="${SCRIPT_DIR}/finetune_qlora.py"
ADAPTER_LOCAL="${SCRIPT_DIR}/carevoice-lora"

echo "=== CareVoice QLoRA — RunPod Training ==="
echo "  Pod    : $POD_ID"
echo "  SSH    : root@${SSH_IP}:${SSH_PORT}"
echo ""

# ── 1. Upload the training script ────────────────────────────────────────────
echo "[1/6] Uploading finetune_qlora.py..."
$SCP "$FINETUNE_PY" "root@${SSH_IP}:/workspace/finetune_qlora.py"

# ── 2. Install deps on the pod ────────────────────────────────────────────────
echo "[2/6] Installing dependencies (transformers from git, peft, trl, bitsandbytes)..."
$SSH bash -s << 'REMOTE_INSTALL'
set -e
pip install -q \
  git+https://github.com/huggingface/transformers.git \
  "peft>=0.14.0" "trl>=0.16.0" "bitsandbytes>=0.43.0" \
  "accelerate>=0.34.0" "datasets>=2.20.0" \
  soundfile librosa requests
echo "Deps installed."
REMOTE_INSTALL

# ── 3. Download Gemma 4 weights from HuggingFace ─────────────────────────────
echo "[3/6] Downloading Gemma 4 4B-IT weights to /workspace/gemma4 (~10 min)..."
$SSH bash -s << 'REMOTE_DOWNLOAD'
set -e
if [ ! -d /workspace/gemma4 ] || [ -z "$(ls -A /workspace/gemma4 2>/dev/null)" ]; then
  pip install -q huggingface_hub
  python3 -c "
from huggingface_hub import snapshot_download
print('Downloading gemma-4-e4b-it...')
snapshot_download(
    repo_id='google/gemma-4-e4b-it',
    local_dir='/workspace/gemma4',
    ignore_patterns=['*.msgpack', '*.h5', 'flax_model*', 'tf_model*'],
)
print('Download complete.')
"
else
  echo "Model already present at /workspace/gemma4"
fi
REMOTE_DOWNLOAD

# ── 4. Patch paths for RunPod and run training ────────────────────────────────
echo "[4/6] Patching paths for RunPod and launching training..."
$SSH bash -s << 'REMOTE_TRAIN'
set -e
cd /workspace

# Patch: replace Kaggle output paths with /workspace paths
sed -i \
  's|ADAPTER_OUT.*=.*"/kaggle/working/carevoice-lora"|ADAPTER_OUT = "/workspace/carevoice-lora"|' \
  finetune_qlora.py
sed -i \
  's|RESULTS_OUT.*=.*"/kaggle/working/finetune_results.json"|RESULTS_OUT = "/workspace/finetune_results.json"|' \
  finetune_qlora.py

# Remove the Kaggle-only P100 fast-exit (RunPod always has a good GPU)
python3 -c "
code = open('finetune_qlora.py').read()
code = code.replace(
    'sys.exit(1)   # fast fail — retry script sees ERROR and pushes next version',
    'pass  # RunPod: skip P100 fast-exit, GPU is always sm_75+'
)
open('finetune_qlora.py', 'w').write(code)
print('Patched P100 exit guard.')
"

echo "Starting training... (estimated 2-4 hours)"
python3 finetune_qlora.py 2>&1 | tee /workspace/finetune_training.log
echo "Training done."
REMOTE_TRAIN

# ── 5. Download adapter ───────────────────────────────────────────────────────
echo "[5/6] Downloading LoRA adapter..."
mkdir -p "$ADAPTER_LOCAL"
$SCP -r "root@${SSH_IP}:/workspace/carevoice-lora/." "$ADAPTER_LOCAL/"
$SCP "root@${SSH_IP}:/workspace/finetune_results.json" "${SCRIPT_DIR}/finetune_results.json" 2>/dev/null || true

echo ""
echo "=== SUCCESS ==="
echo "Adapter saved to: $ADAPTER_LOCAL"
if [ -f "${SCRIPT_DIR}/finetune_results.json" ]; then
  echo "Results:"
  cat "${SCRIPT_DIR}/finetune_results.json"
fi
echo ""
echo "Next step: bash upload_adapter_to_kaggle.sh"

# ── 6. Stop the pod to halt GPU billing ──────────────────────────────────────
echo "[6/6] Stopping pod to stop GPU billing..."
API_KEY="${RUNPOD_API_KEY:?Set RUNPOD_API_KEY env var}"
curl -s --request POST \
  --url "https://api.runpod.io/graphql?api_key=${API_KEY}" \
  --header "Content-Type: application/json" \
  --data "{\"query\": \"mutation { podStop(input: { podId: \\\"${POD_ID}\\\" }) { id desiredStatus } }\"}" \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print('Pod stopped:', d['data']['podStop']['desiredStatus'])" 2>/dev/null || echo "Stop command sent."
