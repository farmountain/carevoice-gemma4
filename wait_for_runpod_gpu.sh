#!/usr/bin/env bash
# wait_for_runpod_gpu.sh
# Polls RunPod every 5 minutes until a GPU is available, then launches pod.
# Tries GPU types in order of preference (best for QLoRA on Gemma 4 4B).

set -e

API_KEY="${RUNPOD_API_KEY:?Set RUNPOD_API_KEY env var}"
GQL="https://api.runpod.io/graphql?api_key=${API_KEY}"
POLL_INTERVAL=300   # 5 minutes

GPU_TYPES=(
  "NVIDIA RTX A6000"
  "NVIDIA GeForce RTX 4090"
  "NVIDIA RTX A5000"
  "NVIDIA GeForce RTX 3090"
  "NVIDIA A100-SXM4-40GB"
  "NVIDIA A100 80GB PCIe"
  "NVIDIA RTX 6000 Ada Generation"
  "NVIDIA L40S"
  "NVIDIA L40"
  "NVIDIA RTX A4500"
  "NVIDIA RTX 4000 Ada Generation"
)

try_launch() {
  local GPU="$1"
  local CLOUD="$2"
  local RESULT
  RESULT=$(curl -s --request POST --url "$GQL" \
    --header "Content-Type: application/json" \
    --data "{\"query\": \"mutation { podFindAndDeployOnDemand(input: { gpuTypeId: \\\"${GPU}\\\", containerDiskInGb: 60, volumeInGb: 0, cloudType: ${CLOUD}, imageName: \\\"runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04\\\", name: \\\"carevoice-finetune\\\" }) { id desiredStatus machine { gpuDisplayName } } }\"}")
  POD_ID=$(echo "$RESULT" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['data']['podFindAndDeployOnDemand']['id'])" 2>/dev/null || echo "")
  if [ -n "$POD_ID" ]; then
    GPU_NAME=$(echo "$RESULT" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['data']['podFindAndDeployOnDemand']['machine']['gpuDisplayName'])" 2>/dev/null || echo "$GPU")
    echo "=== POD LAUNCHED ==="
    echo "  Pod ID : $POD_ID"
    echo "  GPU    : $GPU_NAME"
    echo "  Cloud  : $CLOUD"
    echo "$POD_ID" > /tmp/new_pod_id.txt
    return 0
  fi
  return 1
}

echo "=== RunPod GPU Poller ==="
echo "  Checking every ${POLL_INTERVAL}s for available GPU..."
echo "  GPU preference: ${GPU_TYPES[*]}"
echo ""

ATTEMPT=0
while true; do
  ATTEMPT=$((ATTEMPT + 1))
  echo "--- Poll $ATTEMPT  ($(date '+%H:%M:%S')) ---"

  # Try community cloud first (cheaper), then secure
  for GPU in "${GPU_TYPES[@]}"; do
    for CLOUD in COMMUNITY SECURE; do
      if try_launch "$GPU" "$CLOUD"; then
        echo ""
        echo "Waiting 30s for pod to initialise..."
        sleep 30

        # Get SSH details
        POD_ID=$(cat /tmp/new_pod_id.txt)
        SSH_INFO=$(curl -s --request POST --url "$GQL" \
          --header "Content-Type: application/json" \
          --data "{\"query\": \"{ pod(input: { podId: \\\"${POD_ID}\\\" }) { runtime { ports { ip publicPort type } } } }\"}")
        SSH_IP=$(echo "$SSH_INFO" | python3 -c "import json,sys; ports=json.load(sys.stdin)['data']['pod']['runtime']['ports']; p=[x for x in ports if x['type']=='tcp'][0]; print(p['ip'])" 2>/dev/null || echo "")
        SSH_PORT=$(echo "$SSH_INFO" | python3 -c "import json,sys; ports=json.load(sys.stdin)['data']['pod']['runtime']['ports']; p=[x for x in ports if x['type']=='tcp'][0]; print(p['publicPort'])" 2>/dev/null || echo "")

        if [ -n "$SSH_IP" ]; then
          echo "SSH    : ssh root@${SSH_IP} -p ${SSH_PORT}"
          echo ""
          echo "Next step: bash run_finetune_on_runpod.sh ${POD_ID} ${SSH_IP} ${SSH_PORT}"
        else
          echo "Pod launched but SSH not ready yet. Check: bash runpod.sh status"
          echo "Pod ID: $POD_ID"
        fi
        exit 0
      fi
    done
  done

  echo "  No GPU available yet. Sleeping ${POLL_INTERVAL}s..."
  sleep $POLL_INTERVAL
done
