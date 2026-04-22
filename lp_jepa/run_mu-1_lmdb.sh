#!/bin/bash
# Train mu=-1 on LMDB (pusht_expert_lmdb, 112 JPEG) in the new le-wms env.
# Distinct output name so it doesn't collide with the HDF5 mu-1 sweep run.

set -u
REPO=/home/ubuntu/le-wm-palash
SWEEP_DIR="$REPO/lp_jepa/mu_sweep_pusht_expert"
TRAIN_LOG="$SWEEP_DIR/training_mu-1_lmdb.log"

cd "$REPO"
# shellcheck disable=SC1091
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate le-wms
export STABLEWM_HOME=/home/ubuntu/le-wm-palash/lewm-pusht
set -o pipefail

MODEL_NAME="lewm_lp_p1.0_mu-1_pusht_expert_lmdb"
WANDB_ID="lewm_lp_pusht_expert_mu-1_lmdb"

echo "================================================================"
echo "=== $(date) === START mu=-1  LMDB  env=le-wms  (model=$MODEL_NAME)"
echo "================================================================"

: > "$TRAIN_LOG"
python -u lp_jepa/train_lp_jepa.py \
    --config-name=lewm_lp_jepa_pusht_expert \
    +loss.rdmreg.kwargs.mu=-1.0 \
    num_workers=8 \
    output_model_name=$MODEL_NAME \
    wandb.config.id=$WANDB_ID \
    wandb.config.name=$MODEL_NAME \
    2>&1 | tee "$TRAIN_LOG"
echo "TRAINING_EXITED_${PIPESTATUS[0]}" | tee -a "$TRAIN_LOG"
