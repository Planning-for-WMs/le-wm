#!/bin/bash
# Full pipeline:
#   Phase 0: Convert pusht_expert_train.h5 → pusht_expert_train.lmdb (img_size=112, jpeg q95)
#   Phase 1: Train lp_jepa (p=1.0) on the LMDB with num_workers=6
#   Phase 2: Sweep-eval all epoch checkpoints on PushT-v1
#
# Usage:  tmux new-session -d -s lp_jepa_pusht_expert \
#             bash /home/aarav/wms/le-wm/lp_jepa/run_pusht_lmdb_convert_train_eval.sh

set -u

REPO=/home/aarav/wms/le-wm
CONVERT_LOG="$REPO/lp_jepa/convert_pusht_lmdb.log"
TRAIN_LOG="$REPO/lp_jepa/training_pusht_expert.log"
EVAL_LOG="$REPO/lp_jepa/eval_sweep_pusht_expert.log"

cd "$REPO"
# shellcheck disable=SC1091
source le-wm/bin/activate
set -o pipefail

H5_PATH="$REPO/dataset/pusht_expert_train.h5"
LMDB_PATH="$REPO/dataset/pusht_expert_train.lmdb"

# ═══════════════════════════════════════════════════════
# PHASE 0: HDF5 → LMDB CONVERSION
# ═══════════════════════════════════════════════════════
echo "================================================================"
echo "=== $(date) === PHASE 0: convert HDF5 → LMDB"
echo "================================================================"

if [ -d "$LMDB_PATH" ] && [ -f "$LMDB_PATH/data.mdb" ]; then
    echo "=== $(date) === LMDB already exists at $LMDB_PATH, skipping conversion"
else
    : > "$CONVERT_LOG"
    python convert_to_lmdb.py \
        h5_path="$H5_PATH" \
        out_path="$LMDB_PATH" \
        img_size=112 \
        encoding=jpeg \
        jpeg_quality=95 \
        2>&1 | tee "$CONVERT_LOG"
    CONVERT_EXIT=$?

    if [ $CONVERT_EXIT -ne 0 ] || [ ! -f "$LMDB_PATH/data.mdb" ]; then
        echo "=== $(date) === ERROR: conversion failed (exit $CONVERT_EXIT). Aborting."
        exit 1
    fi
fi

echo "=== $(date) === LMDB ready at $LMDB_PATH"
du -sh "$LMDB_PATH"

# ═══════════════════════════════════════════════════════
# PHASE 1: TRAINING
# ═══════════════════════════════════════════════════════
echo "================================================================"
echo "=== $(date) === PHASE 1: training p=1.0 on LMDB (num_workers=6)"
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

EPOCHS=$(ls "$REPO"/dataset/lewm_lp_p1.0_pusht_expert_epoch_*_object.ckpt 2>/dev/null \
    | grep -oE 'epoch_[0-9]+_object' \
    | grep -oE '[0-9]+' \
    | sort -n \
    | uniq)

if [ -z "$EPOCHS" ]; then
    echo "=== $(date) === ERROR: no epoch checkpoints found. Aborting eval."
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
echo "================================================================"
echo ""
echo "╔══ SUCCESS RATES ══════════════════════════════════════════════╗"
grep -E 'success_rate' "$EVAL_LOG" | head -20
echo "╚══════════════════════════════════════════════════════════════╝"
