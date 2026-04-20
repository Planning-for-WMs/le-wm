"""Minimal LMDBDataset that reads the layout produced by `convert_to_lmdb.py`
and `convert_pusht_to_lmdb.py`. Patches itself onto `swm.data.LMDBDataset` so
existing code (`train_antmaze_explore.py`, configs, etc.) keeps working.

Layout:
  meta:<col>           full numpy buffer (per-frame, length n_frames)
  meta:<col>:shape     int64 shape tuple
  meta:<col>:dtype     dtype string
  n_frames             uint32
  img_size, channels   header bytes
  meta_keys, encoding  comma-separated col names; "jpeg" or "raw"
  <frame_idx>          encoded image bytes (4-byte big-endian frame idx)
"""
import struct
from pathlib import Path

import cv2
import lmdb
import numpy as np
import torch
from einops import rearrange

import stable_worldmodel as swm
from stable_worldmodel.data.dataset import Dataset


class LMDBDataset(Dataset):
    """Reads the LMDB written by convert_to_lmdb / convert_pusht_to_lmdb."""

    def __init__(
        self,
        name: str,
        cache_dir: str | Path,
        num_steps: int = 1,
        frameskip: int = 1,
        keys_to_load: list[str] | None = None,
        keys_to_cache: list[str] | None = None,
        transform=None,
    ):
        path = Path(cache_dir) / name
        if not path.suffix:                # accept both `name` and `name.lmdb`
            for cand in (path, path.with_suffix(".lmdb")):
                if cand.exists():
                    path = cand
                    break
        self.path = path
        self._env = lmdb.open(str(path), readonly=True, lock=False, readahead=False,
                              meminit=False, max_readers=512, subdir=path.is_dir())
        with self._env.begin(write=False) as txn:
            self.n_frames = struct.unpack("<I", txn.get(b"n_frames"))[0]
            self.img_h, self.img_w = struct.unpack("<HH", txn.get(b"img_size"))
            self.channels = struct.unpack("<H", txn.get(b"channels"))[0]
            self.meta_keys_all = txn.get(b"meta_keys").decode().split(",")
            self.encoding = txn.get(b"encoding").decode()

            self._meta = {}
            for k in self.meta_keys_all:
                buf   = txn.get(f"meta:{k}".encode())
                shape = np.frombuffer(txn.get(f"meta:{k}:shape".encode()), dtype=np.int64)
                dtype = np.dtype(txn.get(f"meta:{k}:dtype".encode()).decode())
                self._meta[k] = np.frombuffer(buf, dtype=dtype).reshape(shape).copy()

        assert "ep_len" in self._meta and "ep_offset" in self._meta, \
            f"LMDB at {path} missing ep_len/ep_offset metadata"
        lengths = self._meta["ep_len"].astype(np.int64)
        offsets = self._meta["ep_offset"].astype(np.int64)

        # Default to all data columns if user didn't specify.
        self._keys = list(keys_to_load) if keys_to_load else (
            ["pixels"] + [k for k in self.meta_keys_all if k not in ("ep_len", "ep_offset")]
        )

        super().__init__(lengths, offsets, frameskip=frameskip,
                         num_steps=num_steps, transform=transform)

    # ---- minimal Dataset interface ---------------------------------------------

    @property
    def column_names(self) -> list[str]:
        return self._keys

    def get_dim(self, col: str) -> int:
        if col == "pixels":
            return self.channels * self.img_h * self.img_w
        return int(self._meta[col].shape[-1]) if self._meta[col].ndim > 1 else 1

    def get_col_data(self, col: str) -> np.ndarray:
        if col == "pixels":
            raise NotImplementedError("pixels are decoded lazily, not as a single array")
        return self._meta[col]

    def get_row_data(self, row_idx) -> dict:
        rows = list(row_idx) if isinstance(row_idx, (list, tuple, np.ndarray)) else [row_idx]
        # ep_len / ep_offset are per-episode (length n_eps), the other meta
        # columns are per-frame (length n_frames). Skip the per-episode ones
        # so we don't try to index them with frame rows.
        per_episode = {"ep_len", "ep_offset"}
        out = {
            col: self._meta[col][rows]
            for col in self.meta_keys_all
            if col in self._meta and col not in per_episode
        }
        if "pixels" in self._keys:
            out["pixels"] = np.stack([self._decode_frame(i) for i in rows], axis=0)
        return out

    # ---- frame reading ---------------------------------------------------------

    def _decode_frame(self, frame_idx: int) -> np.ndarray:
        with self._env.begin(write=False) as txn:
            blob = txn.get(struct.pack("<I", int(frame_idx)))
        if self.encoding == "jpeg":
            arr = np.frombuffer(blob, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            return np.frombuffer(blob, dtype=np.uint8).reshape(self.img_h, self.img_w, 3).copy()

    # ---- slice loader (called by Dataset.__getitem__) --------------------------

    def _load_slice(self, ep_idx: int, start: int, end: int) -> dict:
        g_start = int(self.offsets[ep_idx] + start)
        g_end = int(self.offsets[ep_idx] + end)
        steps = {}
        for col in self._keys:
            if col == "pixels":
                pixels = np.stack(
                    [self._decode_frame(i) for i in range(g_start, g_end, self.frameskip)],
                    axis=0,
                )                                                       # (T, H, W, 3) uint8
                t = torch.from_numpy(pixels)
                steps["pixels"] = rearrange(t, "t h w c -> t c h w")    # (T, 3, H, W) uint8
            elif col == "action":
                steps["action"] = torch.from_numpy(self._meta["action"][g_start:g_end].copy())
            else:
                arr = self._meta[col][g_start:g_end:self.frameskip]
                steps[col] = torch.from_numpy(arr.copy())
        return steps


# Inject into swm so existing `swm.data.LMDBDataset(...)` calls work transparently.
swm.data.LMDBDataset = LMDBDataset
