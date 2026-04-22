#!/bin/bash
# Eval sweep for mu=-4 and mu=-3, epochs 8/9/10.
# Runs sequentially in a tmux session alongside the live mu-2 training.

set -u
REPO=/home/ubuntu/le-wm-palash
SWEEP_DIR="$REPO/lp_jepa/mu_sweep_pusht_expert"
cd "$REPO"
# shellcheck disable=SC1091
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate jepa-wms
export STABLEWM_HOME=/home/ubuntu/le-wm-palash/lewm-pusht
set -o pipefail

EPOCHS=(8 9 10)
MUS=(-4 -3)

for MU in "${MUS[@]}"; do
    MU_TAG="mu${MU}"
    MODEL_NAME="lewm_lp_p1.0_${MU_TAG}_pusht_expert"
    EVAL_LOG="$SWEEP_DIR/eval_catchup_${MU_TAG}.log"

    echo "================================================================"
    echo "=== $(date) === EVAL mu=$MU  epochs: ${EPOCHS[*]}"
    echo "================================================================"

    : > "$EVAL_LOG"
    {
        for e in "${EPOCHS[@]}"; do
            CKPT="$REPO/lewm-pusht/${MODEL_NAME}_epoch_${e}_object.ckpt"
            if [ ! -f "$CKPT" ]; then
                echo "=== $(date) === MISSING CKPT $CKPT — skipping"
                continue
            fi
            echo "=== $(date) === EVAL mu=$MU epoch_$e START ==="
            python eval.py \
                policy=${MODEL_NAME}_epoch_$e \
                eval.dataset_name=pusht_expert_train \
                eval.img_size=224 \
                output.filename=lp_jepa_mu_sweep/${MU_TAG}/epoch$e.txt
            echo "=== $(date) === mu=$MU epoch_$e DONE ==="
        done
        echo "EVAL_CATCHUP_COMPLETE_MU_${MU}"
    } 2>&1 | tee -a "$EVAL_LOG"
done

echo "================================================================"
echo "=== $(date) === ALL CATCHUP EVALS DONE"
echo "================================================================"
