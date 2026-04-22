#!/bin/bash
# Train mu=+2 on LMDB (pusht_expert_lmdb), in parallel with the ongoing HDF5 sweep.
# Distinct output name so it doesn't collide with the planned HDF5 mu2 run.

set -u
REPO=/home/ubuntu/le-wm-palash
SWEEP_DIR="$REPO/lp_jepa/mu_sweep_pusht_expert"
TRAIN_LOG="$SWEEP_DIR/training_mu2_lmdb.log"

cd "$REPO"
# shellcheck disable=SC1091
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate jepa-wms
export STABLEWM_HOME=/home/ubuntu/le-wm-palash/lewm-pusht
set -o pipefail

MODEL_NAME="lewm_lp_p1.0_mu2_pusht_expert_lmdb"
WANDB_ID="lewm_lp_pusht_expert_mu2_lmdb"

echo "================================================================"
echo "=== $(date) === START mu=+2  LMDB  (model=$MODEL_NAME)"
echo "================================================================"

: > "$TRAIN_LOG"
python -u lp_jepa/train_lp_jepa.py \
    --config-name=lewm_lp_jepa_pusht_expert \
    +loss.rdmreg.kwargs.mu=2.0 \
    num_workers=4 \
    output_model_name=$MODEL_NAME \
    wandb.config.id=$WANDB_ID \
    wandb.config.name=$MODEL_NAME \
    2>&1 | tee "$TRAIN_LOG"
echo "TRAINING_EXITED_${PIPESTATUS[0]}" | tee -a "$TRAIN_LOG"
