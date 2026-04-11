#!/usr/bin/env bash
# Download the official DINO-WM PushT checkpoint from the OSF mirror.
#
# Usage:
#   bash hierarchical/download_ckpt.sh
#
# After this finishes, the checkpoint lives at:
#   hierarchical/ckpt/outputs/pusht/checkpoints/model_latest.pth
# which is the path expected by config/eval/dino.yaml.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKPT_DIR="$SCRIPT_DIR/ckpt"
ZIP_PATH="$CKPT_DIR/outputs.zip"
TARGET="$CKPT_DIR/outputs/pusht/checkpoints/model_latest.pth"

# OSF anonymous-view-only mirror published by the official DINO-WM repo
# (https://github.com/gaoyuezhou/dino_wm). Bundles checkpoints for
# pusht / point_maze / wall_single in a single 910 MB zip.
OSF_URL="https://osf.io/download/xvzs4/?view_only=a56a296ce3b24cceaf408383a175ce28"

mkdir -p "$CKPT_DIR"

if [[ -f "$TARGET" ]]; then
    echo "[download_ckpt] Already present: $TARGET"
    exit 0
fi

if [[ ! -f "$ZIP_PATH" ]]; then
    echo "[download_ckpt] Fetching outputs.zip (~910 MB) from OSF ..."
    curl -L --fail -o "$ZIP_PATH" "$OSF_URL"
else
    echo "[download_ckpt] outputs.zip already on disk, skipping download."
fi

echo "[download_ckpt] Extracting PushT checkpoint ..."
unzip -o "$ZIP_PATH" "outputs/pusht/*" -d "$CKPT_DIR" >/dev/null

if [[ ! -f "$TARGET" ]]; then
    echo "[download_ckpt] ERROR: extraction did not produce $TARGET" >&2
    exit 1
fi

echo "[download_ckpt] Done. Checkpoint at: $TARGET"
echo "[download_ckpt] You may delete $ZIP_PATH to reclaim ~910 MB."