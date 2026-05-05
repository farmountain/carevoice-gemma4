#!/usr/bin/env bash
# upload_adapter_to_kaggle.sh
# -----------------------------------------------------------------------
# Run AFTER RunPod training completes.
# Downloads LoRA adapter from RunPod via SCP, creates a Kaggle dataset,
# updates inference kernel-metadata.json, and pushes v25.
#
# Usage:
#   bash upload_adapter_to_kaggle.sh <SSH_IP> <SSH_PORT>
#   bash upload_adapter_to_kaggle.sh 157.157.221.29 11052
# -----------------------------------------------------------------------

set -e

SSH_IP="${1:-157.157.221.29}"
SSH_PORT="${2:-11052}"
SSH_KEY="C:/Users/user/.ssh/id_ed25519"
PULL_DIR="C:/Users/user/AppData/Local/Temp/carevoice-lora-pull"
HACKATHON_DIR="D:/all_projects/kaggle_winning_agentic_workflow/solutions/gemma_hackathon"

echo "=== Step 1: Download adapter from RunPod ==="
mkdir -p "$PULL_DIR/carevoice-lora"
scp -o StrictHostKeyChecking=no -i "$SSH_KEY" -P "$SSH_PORT" \
  -r "root@${SSH_IP}:/workspace/carevoice-lora/." "$PULL_DIR/carevoice-lora/"
scp -o StrictHostKeyChecking=no -i "$SSH_KEY" -P "$SSH_PORT" \
  "root@${SSH_IP}:/workspace/finetune_results.json" "$PULL_DIR/" 2>/dev/null || true

ADAPTER_PATH="$PULL_DIR/carevoice-lora"
if [ ! -d "$ADAPTER_PATH" ] || [ -z "$(ls -A $ADAPTER_PATH 2>/dev/null)" ]; then
    echo "ERROR: Adapter directory empty or not found at $ADAPTER_PATH"
    exit 1
fi
echo "  Adapter files:"
ls "$ADAPTER_PATH/"

echo ""
echo "=== Step 2: Finetune results ==="
cat "$PULL_DIR/finetune_results.json" 2>/dev/null || echo "  (no finetune_results.json)"

echo ""
echo "=== Step 3: Create Kaggle dataset metadata ==="
cat > "$ADAPTER_PATH/dataset-metadata.json" << 'META'
{
  "title": "CareVoice QLoRA LoRA Adapter (SurgWound + SPRSound)",
  "id": "farmountain/carevoice-lora",
  "licenses": [{"name": "Apache 2.0"}]
}
META

echo "=== Step 4: Create / version Kaggle dataset ==="
kaggle datasets create -p "$ADAPTER_PATH" --dir-mode zip 2>&1 || \
kaggle datasets version -p "$ADAPTER_PATH" -m "Updated adapter from latest finetune run" 2>&1

echo ""
echo "=== Step 5: Add dataset to inference notebook inputs ==="
python3 - << 'PYEOF'
import json, pathlib
meta_path = pathlib.Path("D:/all_projects/kaggle_winning_agentic_workflow/solutions/gemma_hackathon/kaggle_push/kernel-metadata.json")
meta = json.loads(meta_path.read_text(encoding="utf-8"))
slug = "farmountain/carevoice-lora"
if slug not in meta.get("dataset_sources", []):
    meta.setdefault("dataset_sources", []).append(slug)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Added {slug} to dataset_sources in kernel-metadata.json")
else:
    print(f"  {slug} already present in dataset_sources")
PYEOF

echo ""
echo "=== Step 6: Push v25 inference notebook ==="
cd "$HACKATHON_DIR/kaggle_push"
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 kaggle kernels push 2>&1

echo ""
echo "=== DONE ==="
echo "Monitor v25: kaggle kernels status farmountain/carevoice-gemma4-clinical-intake"
