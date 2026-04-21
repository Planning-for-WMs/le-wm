#!/bin/bash
# Train lp_jepa (p=1.0) on the full pusht_expert_train dataset (44 GB, 18K episodes),
# then sweep-eval all epoch checkpoints on PushT-v1.
#
# Usage:  tmux new-session -d -s lp_jepa_pusht_expert \
#             bash /home/aarav/wms/le-wm/lp_jepa/run_pusht_expert_train_and_eval.sh

set -u

REPO=/home/aarav/wms/le-wm
TRAIN_LOG="$REPO/lp_jepa/training_pusht_expert.log"
EVAL_LOG="$REPO/lp_jepa/eval_sweep_pusht_expert.log"

cd "$REPO"
# shellcheck disable=SC1091
source le-wm/bin/activate
set -o pipefail

# ═══════════════════════════════════════════════════════
# PHASE 1: TRAINING
# ═══════════════════════════════════════════════════════
echo "================================================================"
echo "=== $(date) === PHASE 1: training p=1.0 on pusht_expert_train"
echo "================================================================"

: > "$TRAIN_LOG"
python -u lp_jepa/train_lp_jepa.py \
    --config-name=lewm_lp_jepa_pusht_expert \
    2>&1 | tee "$TRAIN_LOG"
TRAIN_EXIT=$?

echo "TRAINING_EXITED_$TRAIN_EXIT" | tee -a "$TRAIN_LOG"

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "=== $(date) === WARNING: training exited with code $TRAIN_EXIT"
fi

# ═══════════════════════════════════════════════════════
# PHASE 2: EVAL SWEEP
# ═══════════════════════════════════════════════════════
echo "================================================================"
echo "=== $(date) === PHASE 2: eval sweep on PushT-v1"
echo "================================================================"

# Discover checkpoints
EPOCHS=$(ls "$REPO"/dataset/lewm_lp_p1.0_pusht_expert_epoch_*_object.ckpt 2>/dev/null \
    | grep -oE 'epoch_[0-9]+_object' \
    | grep -oE '[0-9]+' \
    | sort -n \
    | uniq)

if [ -z "$EPOCHS" ]; then
    echo "=== $(date) === ERROR: no lewm_lp_p1.0_pusht_expert_epoch_*_object.ckpt found. Aborting eval."
    exit 1
fi

echo "=== $(date) === found epoch checkpoints: $(echo $EPOCHS | tr '\n' ' ')"

: > "$EVAL_LOG"

{
    for e in $EPOCHS; do
        echo "=== $(date) === EVAL epoch_$e START ==="
        python eval.py \
            policy=lewm_lp_p1.0_pusht_expert_epoch_$e \
            eval.dataset_name=pusht_expert_train \
            eval.img_size=224 \
            output.filename=lp_jepa_eval_sweep_pusht_expert/epoch$e.txt
        echo "=== $(date) === epoch_$e DONE ==="
    done
    echo SWEEP_COMPLETE
} 2>&1 | tee -a "$EVAL_LOG"

# ═══════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════
echo "================================================================"
echo "=== $(date) === ALL DONE"
echo "=== training log  : $TRAIN_LOG"
echo "=== eval sweep log: $EVAL_LOG"
echo "================================================================"

# Print success rates for easy scanning
echo ""
echo "╔══ SUCCESS RATES ══════════════════════════════════════════════╗"
grep -E 'success_rate' "$EVAL_LOG" | head -20
echo "╚══════════════════════════════════════════════════════════════╝"
