#!/usr/bin/env bash
# retry_finetune.sh
# -----------------------------------------------------------------------
# Auto-retry the finetune kernel until Kaggle assigns a T4 (sm_75+).
# The kernel (v6+) exits with ERROR in <2 min if P100 (sm_60) assigned,
# writing /kaggle/working/need_t4.json. This script detects ERROR and
# pushes a new version immediately.
#
# Usage:
#   bash retry_finetune.sh [max_retries]   # default: 20
#
# Stops when:
#   - Kernel COMPLETE + finetune_results.json present  -> SUCCESS, prints result
#   - max_retries exhausted                            -> exits 1
#
# After SUCCESS run:
#   bash upload_adapter_to_kaggle.sh
# -----------------------------------------------------------------------

set -e

KERNEL="farmountain/carevoice-qlora-fine-tuning-surgwound-sprsound"
PUSH_DIR="D:/all_projects/kaggle_winning_agentic_workflow/solutions/gemma_hackathon/kaggle_push_finetune"
MAX_RETRIES="${1:-20}"
ATTEMPT=0
CHECK_DIR="/tmp/retry-check-$$"
mkdir -p "$CHECK_DIR"

echo "=== CareVoice QLoRA Finetune Retry Loop ==="
echo "  Kernel : $KERNEL"
echo "  Max    : $MAX_RETRIES retries"
echo ""

push_new_version() {
    cd "$PUSH_DIR"
    PYTHONUTF8=1 PYTHONIOENCODING=utf-8 kaggle kernels push 2>&1 | tail -2
}

check_results() {
    kaggle kernels output "$KERNEL" -p "$CHECK_DIR" --file finetune_results.json 2>/dev/null || true
    [ -f "$CHECK_DIR/finetune_results.json" ]
}

check_need_t4() {
    kaggle kernels output "$KERNEL" -p "$CHECK_DIR" --file need_t4.json 2>/dev/null || true
    [ -f "$CHECK_DIR/need_t4.json" ]
}

get_status() {
    kaggle kernels status "$KERNEL" 2>&1 | grep -oE 'KernelWorkerStatus\.[A-Z]+' | cut -d. -f2 || echo "UNKNOWN"
}

while [ "$ATTEMPT" -lt "$MAX_RETRIES" ]; do
    ATTEMPT=$((ATTEMPT + 1))
    echo "--- Attempt $ATTEMPT / $MAX_RETRIES  ($(date '+%H:%M:%S')) ---"

    push_new_version
    echo "  Pushed v$ATTEMPT. Waiting 3 min for GPU pre-flight..."
    sleep 180   # P100 exits in ~2 min; T4 will still be in pip install at 3 min

    # Poll until terminal state
    while true; do
        STATUS=$(get_status)
        echo "  $(date '+%H:%M:%S')  status=$STATUS"

        case "$STATUS" in
          COMPLETE)
            echo "  COMPLETE — checking for finetune_results.json..."
            if check_results; then
                echo ""
                echo "=== SUCCESS on attempt $ATTEMPT ==="
                cat "$CHECK_DIR/finetune_results.json"
                echo ""
                echo "Next step: bash upload_adapter_to_kaggle.sh"
                exit 0
            else
                echo "  COMPLETE but no results — checking need_t4..."
                if check_need_t4; then
                    GPU=$(python3 -c "import json; d=json.load(open('$CHECK_DIR/need_t4.json')); print(d.get('gpu','?'))" 2>/dev/null || echo "?")
                    echo "  Got $GPU — retrying..."
                fi
                break
            fi
            ;;
          ERROR)
            echo "  ERROR — checking need_t4..."
            if check_need_t4; then
                GPU=$(python3 -c "import json; d=json.load(open('$CHECK_DIR/need_t4.json')); print(d.get('gpu','?'))" 2>/dev/null || echo "?")
                echo "  P100 ($GPU) confirmed — fast exit worked. Retrying..."
            else
                echo "  ERROR without need_t4.json — unexpected. Retrying anyway..."
            fi
            break
            ;;
          RUNNING)
            sleep 120   # check every 2 min while running
            ;;
          *)
            sleep 60
            ;;
        esac
    done

    sleep 15   # brief pause before next push
done

echo "=== EXHAUSTED $MAX_RETRIES retries without T4 ==="
exit 1
