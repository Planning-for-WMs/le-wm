#!/bin/bash
# One-shot HPC script: convert pusht_expert_train.h5 → 224 LMDB,
# then train RDMReg p=1 μ=0 at img_size=224 for 10 epochs,
# then run eval sweep on PushT-v1.
#
# Prerequisites on HPC:
#   - This repo cloned, `cd le-wm`
#   - Python env with stable-worldmodel>=0.0.6, stable-pretraining, lightning,
#     torch, h5py, hdf5plugin, lmdb, opencv-python, hydra-core installed
#   - pusht_expert_train.h5 placed at dataset/pusht_expert_train.h5
#     (download from https://huggingface.co/datasets/quentinll/lewm-pusht)
#
# Usage:
#   bash lp_jepa/run_pusht_224_hpc.sh
#
# Expected runtime on A100-80GB: likely 4-7 hours total (much faster than
# our RTX 4000 Ada's ~15-17 hours due to ~3-4x A100 compute advantage).

set -u
set -o pipefail

REPO=$(cd "$(dirname "$(realpath "$0")")/.." && pwd)
cd "$REPO"

# --- Try common venv activation paths ---
if [ -f "le-wm/bin/activate" ]; then
    source le-wm/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "=== WARNING: no venv found. Using system Python. Adjust as needed."
fi

echo "================================================================"
echo "=== $(date) === ONE-SHOT PUSHT 224 PIPELINE"
echo "=== Python: $(python --version)"
echo "=== GPU:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>&1 | head -1
echo "================================================================"

# ═══════════════════════════════════════════════════════
# PHASE 0: HDF5 → 224 LMDB (skip if already done)
# ═══════════════════════════════════════════════════════
LMDB_PATH="$REPO/dataset/pusht_expert_train_224.lmdb"
H5_PATH="$REPO/dataset/pusht_expert_train.h5"

echo ""
echo "=== $(date) === PHASE 0: HDF5 → 224 LMDB"

if [ -f "$LMDB_PATH/data.mdb" ]; then
    echo "=== LMDB already exists at $LMDB_PATH ($(du -sh $LMDB_PATH | cut -f1)). Skipping conversion."
else
    if [ ! -f "$H5_PATH" ]; then
        echo "=== ERROR: $H5_PATH not found."
        echo "=== Download pusht_expert_train.h5 from"
        echo "===   https://huggingface.co/datasets/quentinll/lewm-pusht"
        echo "=== and place it at $H5_PATH"
        exit 1
    fi
    python convert_to_lmdb.py \
        h5_path="$H5_PATH" \
        out_path="$LMDB_PATH" \
        img_size=224 \
        encoding=jpeg \
        jpeg_quality=95
    echo "=== $(date) === LMDB conversion done. Size: $(du -sh $LMDB_PATH | cut -f1)"
fi

# ═══════════════════════════════════════════════════════
# PHASE 1: TRAINING (RDMReg p=1, μ=0, img=224, 10 epochs)
# ═══════════════════════════════════════════════════════
echo ""
echo "=== $(date) === PHASE 1: training RDMReg p=1 μ=0 at img_size=224"

TRAIN_LOG="$REPO/lp_jepa/training_pusht_expert_224_p1mu0_hpc.log"
: > "$TRAIN_LOG"

python -u lp_jepa/train_lp_jepa.py \
    --config-name=lewm_lp_jepa_pusht_expert_224 \
    2>&1 | tee "$TRAIN_LOG"
TRAIN_EXIT=$?
echo "TRAINING_EXITED_$TRAIN_EXIT" | tee -a "$TRAIN_LOG"

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "=== $(date) === WARNING: training exit code $TRAIN_EXIT"
fi

# ═══════════════════════════════════════════════════════
# PHASE 2: EVAL SWEEP (all epoch checkpoints, eval at 224)
# ═══════════════════════════════════════════════════════
echo ""
echo "=== $(date) === PHASE 2: eval sweep on PushT-v1 at img_size=224"

EVAL_LOG="$REPO/lp_jepa/eval_sweep_pusht_expert_224_p1mu0_hpc.log"

EPOCHS=$(ls "$REPO"/dataset/lewm_lp_p1.0_mu0_pusht_expert_224_epoch_*_object.ckpt 2>/dev/null \
    | grep -oE 'epoch_[0-9]+_object' \
    | grep -oE '[0-9]+' \
    | sort -n \
    | uniq)

if [ -z "$EPOCHS" ]; then
    echo "=== $(date) === ERROR: no epoch checkpoints found; training likely failed."
    exit 1
fi

echo "=== found epoch checkpoints: $(echo $EPOCHS | tr '\n' ' ')"

: > "$EVAL_LOG"
{
    for e in $EPOCHS; do
        echo "=== $(date) === EVAL epoch_$e START ==="
        python eval.py \
            policy=lewm_lp_p1.0_mu0_pusht_expert_224_epoch_$e \
            eval.dataset_name=pusht_expert_train \
            eval.img_size=224 \
            output.filename=lp_jepa_eval_sweep_pusht_expert_224_p1mu0_hpc/epoch$e.txt
        echo "=== $(date) === epoch_$e DONE ==="
    done
    echo SWEEP_COMPLETE
} 2>&1 | tee -a "$EVAL_LOG"

echo ""
echo "================================================================"
echo "=== $(date) === ALL DONE"
echo "================================================================"
echo ""
echo "╔══ SUCCESS RATES ══════════════════════════════════════════════╗"
grep -E 'success_rate' "$EVAL_LOG" | head -20
echo "╚══════════════════════════════════════════════════════════════╝"
