#!/usr/bin/env bash
# Download the original DINO-WM PushT dataset (pusht_noise) from OSF and
# convert it to an LMDB at img_size=112 ready for training.
#
# Usage:
#   bash download_pusht.sh                     # default img_size=112
#   IMG_SIZE=224 bash download_pusht.sh        # override
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/dataset"
IMG_SIZE="${IMG_SIZE:-112}"
PUSHT_DIR="$DATA_DIR/pusht_noise"
ZIP_PATH="$DATA_DIR/pusht_noise.zip"
OSF_URL="https://osf.io/download/678ac7164c237c16bedd6129/?view_only=a56a296ce3b24cceaf408383a175ce28"

mkdir -p "$DATA_DIR"

if [[ ! -d "$PUSHT_DIR" ]]; then
    if [[ ! -f "$ZIP_PATH" ]]; then
        echo "[1/3] Downloading pusht_noise.zip (~2.6 GB)..."
        curl -L --fail -o "$ZIP_PATH" "$OSF_URL"
    fi
    echo "[2/3] Extracting..."
    (cd "$DATA_DIR" && unzip -q pusht_noise.zip && rm -f pusht_noise.zip)
else
    echo "[1-2/3] $PUSHT_DIR already present, skipping download/extract."
fi

for SPLIT in train val; do
    OUT="$DATA_DIR/pusht_${SPLIT}_${IMG_SIZE}.lmdb"
    if [[ -d "$OUT" ]]; then
        echo "[3/3] $OUT already present, skipping."
    else
        echo "[3/3] Converting $SPLIT split -> $OUT (img_size=$IMG_SIZE) ..."
        python "$SCRIPT_DIR/convert_pusht_to_lmdb.py" \
            src="$PUSHT_DIR/$SPLIT" out="$OUT" img_size="$IMG_SIZE"
    fi
done

echo "Done. LMDBs at $DATA_DIR/pusht_{train,val}_${IMG_SIZE}.lmdb"
