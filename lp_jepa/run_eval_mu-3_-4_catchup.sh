#!/bin/bash
# Fill in missing epochs (1..7) for mu=-3 and mu=-4 HDF5 eval sweeps.
# Epochs 8/9/10 already present from earlier catchup.

set -u
REPO=/home/ubuntu/le-wm-palash
SWEEP_DIR="$REPO/lp_jepa/mu_sweep_pusht_expert"
cd "$REPO"
# shellcheck disable=SC1091
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate le-wms
export STABLEWM_HOME=/home/ubuntu/le-wm-palash/lewm-pusht
set -o pipefail

EPOCHS=$(seq 1 7)
MUS=(-3 -4)

for MU in "${MUS[@]}"; do
    MU_TAG="mu${MU}"
    MODEL_NAME="lewm_lp_p1.0_${MU_TAG}_pusht_expert"
    EVAL_LOG="$SWEEP_DIR/eval_sweep_${MU_TAG}.log"

    echo "================================================================"
    echo "=== $(date) === FILL EVAL $MU_TAG  epochs 1..7 (HDF5)"
    echo "================================================================"

    : > "$EVAL_LOG"
    {
        for e in $EPOCHS; do
            CKPT="$REPO/lewm-pusht/${MODEL_NAME}_epoch_${e}_object.ckpt"
            if [ ! -f "$CKPT" ]; then
                echo "=== $(date) === MISSING CKPT $CKPT — skipping"
                continue
            fi
            echo "=== $(date) === EVAL $MU_TAG epoch_$e START ==="
            python eval.py \
                policy=${MODEL_NAME}_epoch_$e \
                eval.dataset_name=pusht_expert_train \
                eval.img_size=224 \
                output.filename=lp_jepa_mu_sweep/${MU_TAG}/epoch$e.txt
            echo "=== $(date) === $MU_TAG epoch_$e DONE ==="
        done
        echo "EVAL_FILL_COMPLETE_${MU_TAG}"
    } 2>&1 | tee -a "$EVAL_LOG"
done

echo "================================================================"
echo "=== $(date) === ALL FILL EVALS DONE"
echo "================================================================"
