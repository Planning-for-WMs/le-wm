#!/bin/bash
# Wait for the in-progress lp_jepa/train_lp_jepa run to finish, then sweep
# eval over all lewm_lp_p1.0_epoch_*_object.ckpt checkpoints using the
# tworoom_lmdb eval config (matches training at img_size=112 on swm/TwoRoom-v1).
#
# Usage:  tmux new-session -d -s lp_jepa_tworoom_chain \
#             bash /home/aarav/wms/le-wm/lp_jepa/run_tworoom_eval_after_train.sh
#
# All output is tee'd to chain_eval.log for post-hoc inspection.

set -u

REPO=/home/aarav/wms/le-wm
TRAIN_LOG="$REPO/lp_jepa/training_tworoom_resume.log"
EVAL_LOG="$REPO/lp_jepa/eval_sweep_tworoom.log"
CHAIN_LOG="$REPO/lp_jepa/chain_eval.log"

# Mirror everything this script does into CHAIN_LOG
exec > >(tee -a "$CHAIN_LOG") 2>&1

echo "================================================================"
echo "=== $(date) === chain script PID $$ started"
echo "=== train log: $TRAIN_LOG"
echo "=== eval log : $EVAL_LOG"
echo "================================================================"

# Poll every 30s until no python training process is running.
echo "=== $(date) === waiting for training to finish (polling every 30s)"
while pgrep -f 'python -u lp_jepa/train_lp_jepa' > /dev/null; do
    sleep 30
done
echo "=== $(date) === no train_lp_jepa process detected"

# Brief settle time for final checkpoint flush
sleep 5

# Check training exit status
if grep -q 'TRAINING_EXITED_0' "$TRAIN_LOG" 2>/dev/null; then
    echo "=== $(date) === training exited cleanly (TRAINING_EXITED_0 found)"
elif grep -qE 'TRAINING_EXITED_[0-9]+' "$TRAIN_LOG" 2>/dev/null; then
    code=$(grep -oE 'TRAINING_EXITED_[0-9]+' "$TRAIN_LOG" | tail -1)
    echo "=== $(date) === WARNING: training did not exit 0 ($code). Proceeding with eval anyway."
else
    echo "=== $(date) === WARNING: no TRAINING_EXITED marker in $TRAIN_LOG — training may have been killed. Proceeding with eval using whatever checkpoints exist."
fi

cd "$REPO"
# shellcheck disable=SC1091
source le-wm/bin/activate
set -o pipefail

# Auto-discover epoch checkpoints so this works whether training finished
# 8, 9, 10, or any other number of epochs.
EPOCHS=$(ls "$REPO"/dataset/lewm_lp_p1.0_epoch_*_object.ckpt 2>/dev/null \
    | grep -oE 'epoch_[0-9]+_object' \
    | grep -oE '[0-9]+' \
    | sort -n \
    | uniq)

if [ -z "$EPOCHS" ]; then
    echo "=== $(date) === ERROR: no lewm_lp_p1.0_epoch_*_object.ckpt files found. Aborting."
    exit 1
fi

echo "=== $(date) === found epoch checkpoints: $(echo $EPOCHS | tr '\n' ' ')"

# Truncate eval log
: > "$EVAL_LOG"

# Run the eval sweep — tworoom_lmdb config matches training env + resolution
{
    echo "=== $(date) === starting tworoom_lmdb eval sweep"
    for e in $EPOCHS; do
        echo "=== $(date) === EVAL epoch_$e START ==="
        python eval.py \
            --config-name=tworoom_lmdb \
            policy=lewm_lp_p1.0_epoch_$e \
            output.filename=lp_jepa_eval_sweep_tworoom/epoch$e.txt
        echo "=== $(date) === epoch_$e DONE ==="
    done
    echo SWEEP_COMPLETE
} 2>&1 | tee -a "$EVAL_LOG"

echo "=== $(date) === chain script finished"
echo "================================================================"
