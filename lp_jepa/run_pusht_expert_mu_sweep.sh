#!/bin/bash
# Sweep mu in {-4,-3,-2,-1,1,2} (skipping mu=0 — already done).
# For each mu: train lp_jepa p=1.0 on pusht_expert_train, then sweep-eval all epoch checkpoints on PushT-v1.
#
# Usage:  tmux new-session -d -s lp_mu_sweep_pusht_expert \
#             bash /home/aarav/wms/le-wm/lp_jepa/run_pusht_expert_mu_sweep.sh

set -u

REPO=/home/aarav/wms/le-wm
SWEEP_DIR="$REPO/lp_jepa/mu_sweep_pusht_expert"
mkdir -p "$SWEEP_DIR"

cd "$REPO"
# shellcheck disable=SC1091
source le-wm/bin/activate
set -o pipefail

MU_VALUES=(-4 -3 -2 -1 1 2)

echo "================================================================"
echo "=== $(date) === mu sweep START: mu in ${MU_VALUES[*]}"
echo "================================================================"

for MU in "${MU_VALUES[@]}"; do
    MU_TAG="mu${MU}"
    MODEL_NAME="lewm_lp_p1.0_${MU_TAG}_pusht_expert"
    TRAIN_LOG="$SWEEP_DIR/training_${MU_TAG}.log"
    EVAL_LOG="$SWEEP_DIR/eval_sweep_${MU_TAG}.log"
    WANDB_ID="lewm_lp_pusht_expert_${MU_TAG}"

    echo "================================================================"
    echo "=== $(date) === START mu=$MU   (model=$MODEL_NAME)"
    echo "================================================================"

    # ─── PHASE 1: train ───────────────────────────────────────────────
    : > "$TRAIN_LOG"
    python -u lp_jepa/train_lp_jepa.py \
        --config-name=lewm_lp_jepa_pusht_expert \
        loss.rdmreg.kwargs.mu=$MU \
        output_model_name=$MODEL_NAME \
        wandb.config.id=$WANDB_ID \
        wandb.config.name=$MODEL_NAME \
        2>&1 | tee "$TRAIN_LOG"
    TRAIN_EXIT=${PIPESTATUS[0]}
    echo "TRAINING_EXITED_$TRAIN_EXIT" | tee -a "$TRAIN_LOG"

    if [ "$TRAIN_EXIT" -ne 0 ]; then
        echo "=== $(date) === WARNING: mu=$MU training exited $TRAIN_EXIT — skipping eval, moving to next mu"
        continue
    fi

    # ─── PHASE 2: eval sweep ──────────────────────────────────────────
    EPOCHS=$(ls "$REPO"/dataset/${MODEL_NAME}_epoch_*_object.ckpt 2>/dev/null \
        | grep -oE 'epoch_[0-9]+_object' \
        | grep -oE '[0-9]+' \
        | sort -n \
        | uniq)

    if [ -z "$EPOCHS" ]; then
        echo "=== $(date) === ERROR: no checkpoints for mu=$MU (pattern ${MODEL_NAME}_epoch_*_object.ckpt) — skipping eval"
        continue
    fi

    echo "=== $(date) === mu=$MU found epoch checkpoints: $(echo $EPOCHS | tr '\n' ' ')"

    : > "$EVAL_LOG"
    {
        for e in $EPOCHS; do
            echo "=== $(date) === EVAL mu=$MU epoch_$e START ==="
            python eval.py \
                policy=${MODEL_NAME}_epoch_$e \
                eval.dataset_name=pusht_expert_train \
                eval.img_size=224 \
                output.filename=lp_jepa_mu_sweep/${MU_TAG}/epoch$e.txt
            echo "=== $(date) === mu=$MU epoch_$e DONE ==="
        done
        echo "SWEEP_COMPLETE_MU_${MU}"
    } 2>&1 | tee -a "$EVAL_LOG"

    echo "=== $(date) === mu=$MU COMPLETE (train + eval)"
done

# ═══════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════
echo ""
echo "================================================================"
echo "=== $(date) === FULL SWEEP DONE"
echo "================================================================"
echo ""
echo "╔══ BEST SUCCESS RATE PER MU ═══════════════════════════════════╗"
for MU in "${MU_VALUES[@]}"; do
    LOG="$SWEEP_DIR/eval_sweep_mu${MU}.log"
    if [ -f "$LOG" ]; then
        BEST=$(grep -oE "'success_rate': [0-9.]+" "$LOG" | awk -F': ' '{print $2}' | sort -g | tail -1)
        printf "  mu=%2d  best_success_rate=%s\n" "$MU" "${BEST:-n/a}"
    else
        printf "  mu=%2d  (no eval log)\n" "$MU"
    fi
done
echo "╚══════════════════════════════════════════════════════════════╝"
