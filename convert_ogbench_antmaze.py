"""Convert an OGBench locomaze dataset to the le-wm LMDB format.

OGBench ships locomaze datasets (e.g. ``antmaze-medium-explore-v0``) as a single
``.npz`` file containing 29-d MuJoCo states (``qpos + qvel``), 8-d actions, and
terminal flags. LeWM trains on pixels, so this script reconstructs frames from
the saved states via ``env.set_state(qpos, qvel); env.render(...)`` and writes
them to an LMDB that the existing ``LMDBDataset`` loader can consume unchanged.

The output schema matches ``convert_to_lmdb.py``:
  - per-frame JPEG (or raw) bytes keyed by global frame index (little-endian u32)
  - ``meta:ep_len``, ``meta:ep_offset``, ``meta:action``, ``meta:observation``
  - headers: ``n_frames``, ``img_size``, ``channels``, ``encoding``, ``meta_keys``

Usage:
    MUJOCO_GL=egl python convert_ogbench_antmaze.py --max_episodes 200
"""

import argparse
import os
import struct
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import lmdb
import numpy as np
from tqdm import tqdm

import ogbench


# --- Ant joint layout (from ogbench/locomaze/ant.py: get_ob concatenates qpos+qvel)
ANT_NQ = 15
ANT_NV = 14
ANT_OBS_DIM = ANT_NQ + ANT_NV  # 29


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset_name", default="antmaze-medium-explore-v0",
                   help="OGBench dataset id (the env id without the split suffix is used for rendering).")
    p.add_argument("--out_path", default="dataset/antmaze_medium_explore.lmdb",
                   help="Output LMDB directory.")
    p.add_argument("--ogbench_data_dir", default="~/.ogbench/data",
                   help="Where to cache the downloaded OGBench npz.")
    p.add_argument("--img_size", type=int, default=112, help="Rendered frame size (square).")
    p.add_argument("--jpeg_quality", type=int, default=95, help="JPEG quality (1-100).")
    p.add_argument("--encoding", choices=["jpeg", "raw"], default="jpeg",
                   help="Pixel encoding in LMDB (jpeg=compressed, raw=uncompressed).")
    p.add_argument("--camera_name", default=None,
                   help="MuJoCo camera name. Leave unset to use ogbench's built-in "
                        "top-down maze camera (elevation=-90, framed on the full maze).")
    p.add_argument("--max_episodes", type=int, default=200,
                   help="Convert only the first N episodes. Set to 0 or negative to convert all.")
    p.add_argument("--include_val", action="store_true",
                   help="Also append the validation split after the train split.")
    return p.parse_args()


def episodes_from_valids(valids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build ep_offset / ep_len arrays as the maximal runs of ``valids==1``.

    In OGBench's compact format, ``valids[t]==0`` marks a trajectory-boundary
    frame whose state is real but whose "next" state crosses into a different
    episode. Every episode is a maximal run of ``valids==1`` frames sitting
    between those boundaries.
    """
    mask = valids.astype(bool)
    # Transition points (False->True = episode start, True->False = episode end).
    padded = np.concatenate([[False], mask, [False]])
    diff = np.diff(padded.astype(np.int8))
    starts = np.nonzero(diff == 1)[0]
    ends = np.nonzero(diff == -1)[0]
    assert len(starts) == len(ends), "valids run detection failed"
    ep_offset = starts.astype(np.int64)
    ep_len = (ends - starts).astype(np.int64)
    return ep_offset, ep_len


def encode_frame(frame: np.ndarray, encoding: str, jpeg_quality: int) -> bytes:
    if encoding == "jpeg":
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        ok, buf = cv2.imencode(".jpg", bgr, params)
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        return buf.tobytes()
    return frame.tobytes()


def estimate_map_size(n_frames: int, img_size: int, encoding: str,
                      action_dim: int, obs_dim: int) -> int:
    if encoding == "jpeg":
        # Scale with img_size^2 so the estimate stays correct at higher
        # resolutions. Empirically JPEG q=95 on MuJoCo renders lands around
        # ~0.5 bytes per raw BGR byte (~6x compression): ~6 KB at 112^2,
        # ~25 KB at 224^2. The previous hardcoded 10_000 B over-shot at
        # 112 but under-shot 224 by ~2x and silently truncated the LMDB
        # mid-conversion when it hit MDB_MAP_FULL.
        per_frame = int(img_size * img_size * 0.5)
    else:
        per_frame = img_size * img_size * 3
    per_meta = (action_dim + obs_dim) * 4  # float32
    return int(n_frames * (per_frame + per_meta) * 1.5) + 512 * 1024 * 1024


def main():
    args = parse_args()

    # --- Make the env (renders pixels from state). The dataset-side suffix
    # (-explore-v0 / -navigate-v0 / ...) is only used to pick the npz file;
    # make_env_and_datasets strips it to get the registered env id.
    print(f"Building env for {args.dataset_name} ...")
    env_kwargs = dict(
        render_mode="rgb_array",
        width=args.img_size,
        height=args.img_size,
    )
    if args.camera_name is not None:
        env_kwargs["camera_name"] = args.camera_name
    env = ogbench.make_env_and_datasets(
        args.dataset_name,
        env_only=True,
        **env_kwargs,
    )
    env.reset()
    nq = env.unwrapped.model.nq
    nv = env.unwrapped.model.nv
    assert nq + nv == ANT_OBS_DIM, (
        f"Expected ant nq+nv={ANT_OBS_DIM} but got nq={nq} nv={nv}. "
        "Are you sure this is an antmaze dataset?"
    )

    # --- Load the dataset (and optionally the -val split).
    print(f"Loading dataset {args.dataset_name} ...")
    train_ds, val_ds = ogbench.make_env_and_datasets(
        args.dataset_name,
        dataset_dir=os.path.expanduser(args.ogbench_data_dir),
        compact_dataset=True,
        dataset_only=True,
        cur_env=env,
    )

    splits = [train_ds]
    if args.include_val:
        splits.append(val_ds)

    obs_parts, act_parts, val_parts = [], [], []
    for ds in splits:
        obs_parts.append(ds["observations"])
        act_parts.append(ds["actions"])
        val_parts.append(ds["valids"])
    observations = np.concatenate(obs_parts, axis=0)
    actions = np.concatenate(act_parts, axis=0)
    valids = np.concatenate(val_parts, axis=0)

    src_offset, src_len = episodes_from_valids(valids)
    n_eps_total = len(src_len)
    print(f"Found {n_eps_total} episodes, {int(src_len.sum())} valid steps "
          f"({len(observations)} total raw rows).")

    # --- Subset selection
    if args.max_episodes and args.max_episodes > 0:
        n_keep = min(args.max_episodes, n_eps_total)
    else:
        n_keep = n_eps_total
    print(f"Converting first {n_keep} episodes.")

    src_offset = src_offset[:n_keep]
    src_len = src_len[:n_keep]
    n_frames = int(src_len.sum())

    # Gather valid rows (drop valids==0 boundary frames) so storage is contiguous.
    row_idx = np.concatenate([
        np.arange(s, s + l) for s, l in zip(src_offset, src_len)
    ])
    observations = observations[row_idx]
    actions = actions[row_idx]

    # Rewrite ep_offset to be contiguous in the new packed layout.
    ep_len = src_len.astype(np.int64)
    ep_offset = np.concatenate([[0], np.cumsum(ep_len[:-1])]).astype(np.int64)

    # --- Ensure action has the frame-aligned length le-wm expects.
    assert actions.shape[0] == n_frames, (
        f"action/obs length mismatch: actions={actions.shape[0]}, frames={n_frames}"
    )
    action_dim = int(actions.shape[-1])
    obs_dim = int(observations.shape[-1])

    # --- Open LMDB.
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        print(f"Warning: {out_path} already exists, will overwrite contents.")
    map_size = estimate_map_size(n_frames, args.img_size, args.encoding, action_dim, obs_dim)
    env_lmdb = lmdb.open(str(out_path), map_size=map_size)

    # --- Write headers + metadata.
    meta_arrays = {
        "ep_len": ep_len.astype(np.int64),
        "ep_offset": ep_offset.astype(np.int64),
        "action": actions.astype(np.float32),
        "observation": observations.astype(np.float32),
    }
    with env_lmdb.begin(write=True) as txn:
        for k, v in meta_arrays.items():
            txn.put(f"meta:{k}".encode(), v.tobytes())
            txn.put(f"meta:{k}:shape".encode(),
                    np.array(v.shape, dtype=np.int64).tobytes())
            txn.put(f"meta:{k}:dtype".encode(), str(v.dtype).encode())

        txn.put(b"n_frames", struct.pack("<I", n_frames))
        txn.put(b"img_size", struct.pack("<HH", args.img_size, args.img_size))
        txn.put(b"channels", struct.pack("<H", 3))
        txn.put(b"meta_keys", ",".join(meta_arrays.keys()).encode())
        txn.put(b"encoding", args.encoding.encode())

    # --- Render + write frames.
    print(f"Rendering {n_frames} frames at {args.img_size}x{args.img_size} ({args.encoding}) ...")
    total_bytes = 0
    batch_write_every = 1024
    buf = []

    global_idx = 0
    with tqdm(total=n_frames, desc="Rendering") as pbar:
        for t in range(n_frames):
            state = observations[t]
            qpos = state[:nq].astype(np.float64)
            qvel = state[nq:nq + nv].astype(np.float64)
            env.unwrapped.set_state(qpos, qvel)
            frame = env.render()  # (H, W, 3) uint8
            encoded = encode_frame(frame, args.encoding, args.jpeg_quality)
            buf.append((global_idx, encoded))
            total_bytes += len(encoded)
            global_idx += 1
            pbar.update(1)

            if len(buf) >= batch_write_every:
                with env_lmdb.begin(write=True) as txn:
                    for idx, enc in buf:
                        txn.put(struct.pack("<I", idx), enc)
                buf.clear()

    if buf:
        with env_lmdb.begin(write=True) as txn:
            for idx, enc in buf:
                txn.put(struct.pack("<I", idx), enc)
        buf.clear()

    env_lmdb.close()
    env.close()

    actual_size = sum(p.stat().st_size for p in out_path.iterdir())
    print(f"Done. {global_idx} frames, avg {total_bytes / max(global_idx, 1):.0f} bytes/frame.")
    print(f"LMDB size: {actual_size / 1e9:.2f} GB at {out_path}")


if __name__ == "__main__":
    main()
