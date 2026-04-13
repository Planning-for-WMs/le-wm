"""Convert one split of `pusht_noise/{train,val}` (the original DINO-WM
PushT dataset: per-episode mp4 + concatenated .pth files) into the LMDB
layout that `convert_to_lmdb.py` produces, so the existing LMDBDataset
loader can consume it unchanged.

Run via the shell helper (`download_pusht.sh`) or directly:
    python convert_pusht_to_lmdb.py src=dataset/pusht_noise/train \
                                    out=dataset/pusht_train_112.lmdb \
                                    img_size=112
"""
import pickle
import struct
import sys
from pathlib import Path

import cv2
import decord
import lmdb
import numpy as np
import torch
from tqdm import tqdm

decord.bridge.set_bridge("native")


def main():
    kw = dict(arg.split("=", 1) for arg in sys.argv[1:] if "=" in arg)
    src = Path(kw["src"])
    out = Path(kw["out"])
    img_size = int(kw.get("img_size", 112))
    encoding = kw.get("encoding", "jpeg")
    jpeg_quality = int(kw.get("jpeg_quality", 90))

    # --- Load per-episode metadata.
    states     = torch.load(src / "states.pth").float()
    velocities = torch.load(src / "velocities.pth").float()
    actions    = torch.load(src / "rel_actions.pth").float()           # env-space
    with open(src / "seq_lengths.pkl", "rb") as f:
        seq_lengths = list(pickle.load(f))
    n_ep = len(seq_lengths)
    # Full state (matches swm/PushT-v1 _get_obs):
    # [agent_x, agent_y, T_x, T_y, angle, agent_vx, agent_vy]
    full_states = torch.cat([states, velocities], dim=-1)
    proprios = torch.cat([states[..., :2], velocities], dim=-1)        # 4-d agent only

    n_frames = sum(seq_lengths)
    ep_lens = np.asarray(seq_lengths, dtype=np.int64)
    ep_offsets = np.zeros(n_ep, dtype=np.int64)
    ep_offsets[1:] = np.cumsum(ep_lens[:-1])

    print(f"src={src} -> out={out}")
    print(f"  episodes={n_ep}  total_frames={n_frames}  img_size={img_size} ({encoding})")

    # --- Stack flat per-frame metadata arrays (matches convert_to_lmdb format).
    flat_meta = {}
    for col, src_arr in [("action", actions), ("observation", proprios), ("state", full_states)]:
        flat = []
        for ep in range(n_ep):
            flat.append(src_arr[ep, : ep_lens[ep]].numpy())
        flat_meta[col] = np.concatenate(flat, axis=0)
    flat_meta["ep_len"] = ep_lens
    flat_meta["ep_offset"] = ep_offsets

    # --- Open LMDB, write metadata.
    if encoding == "jpeg":
        per_frame_estimate = 12_000
    else:
        per_frame_estimate = img_size * img_size * 3
    map_size = int(n_frames * per_frame_estimate * 1.4) + 256 * 1024 * 1024
    env = lmdb.open(str(out), map_size=map_size)

    with env.begin(write=True) as txn:
        for k, v in flat_meta.items():
            v = np.ascontiguousarray(v)
            txn.put(f"meta:{k}".encode(), v.tobytes())
            txn.put(f"meta:{k}:shape".encode(),
                    np.array(v.shape, dtype=np.int64).tobytes())
            txn.put(f"meta:{k}:dtype".encode(), str(v.dtype).encode())
        txn.put(b"n_frames", struct.pack("<I", n_frames))
        txn.put(b"img_size", struct.pack("<HH", img_size, img_size))
        txn.put(b"channels", struct.pack("<H", 3))
        txn.put(b"meta_keys", ",".join(flat_meta.keys()).encode())
        txn.put(b"encoding", encoding.encode())

    # --- Iterate episodes, decode frames from mp4, write each frame.
    total_bytes = 0
    with env.begin(write=True) as txn:
        global_idx = 0
        for ep in tqdm(range(n_ep), desc=f"episodes ({src.name})"):
            mp4 = src / "obses" / f"episode_{ep:03d}.mp4"
            reader = decord.VideoReader(str(mp4), num_threads=1)
            frames = reader.get_batch(list(range(ep_lens[ep]))).asnumpy()  # (T, H, W, 3) uint8
            for t in range(ep_lens[ep]):
                resized = cv2.resize(frames[t], (img_size, img_size),
                                     interpolation=cv2.INTER_AREA)
                if encoding == "jpeg":
                    bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
                    _, buf = cv2.imencode(
                        ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
                    )
                    blob = buf.tobytes()
                else:
                    blob = resized.tobytes()
                txn.put(struct.pack("<I", global_idx), blob)
                total_bytes += len(blob)
                global_idx += 1
    env.close()

    actual = sum(p.stat().st_size for p in out.iterdir())
    print(f"  avg frame size: {total_bytes / n_frames:.0f} B   "
          f"lmdb size: {actual / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
