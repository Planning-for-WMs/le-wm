"""Convert any HDF5 dataset to LMDB with configurable image encoding."""

import struct
from pathlib import Path

import cv2
import h5py
import hdf5plugin  # noqa: F401
import hydra
import lmdb
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm


def estimate_map_size(n_frames: int, cfg: DictConfig) -> int:
    if cfg.encoding == "jpeg":
        per_frame = 10_000  # ~8KB avg for JPEG at typical sizes
    else:
        per_frame = cfg.img_size * cfg.img_size * 3
    return int(n_frames * per_frame * 1.3) + 512 * 1024 * 1024


def encode_frame(frame: np.ndarray, cfg: DictConfig) -> bytes:
    resized = cv2.resize(frame, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_AREA)
    if cfg.encoding == "jpeg":
        params = [int(cv2.IMWRITE_JPEG_QUALITY), cfg.jpeg_quality]
        _, buf = cv2.imencode(".jpg", cv2.cvtColor(resized, cv2.COLOR_RGB2BGR), params)
        return buf.tobytes()
    else:
        return resized.tobytes()


@hydra.main(version_base=None, config_path="./config/convert", config_name="default")
def main(cfg: DictConfig):
    h5_path = Path(cfg.h5_path)
    out_path = Path(cfg.out_path)

    with h5py.File(h5_path, "r") as f:
        n_frames = f[cfg.pixel_key].shape[0]
        h, w, c = f[cfg.pixel_key].shape[1:]
        print(f"Source: {n_frames} frames, {h}x{w}x{c}")

        meta_keys = [k for k in f.keys() if k != cfg.pixel_key]
        meta = {k: f[k][:] for k in meta_keys}
        print(f"Metadata keys: {meta_keys}")

    map_size = estimate_map_size(n_frames, cfg)
    print(f"Target: {cfg.img_size}x{cfg.img_size} {cfg.encoding}" +
          (f" q{cfg.jpeg_quality}" if cfg.encoding == "jpeg" else ""))
    print(f"Writing to {out_path}")

    env = lmdb.open(str(out_path), map_size=map_size)

    with env.begin(write=True) as txn:
        for k, v in meta.items():
            txn.put(f"meta:{k}".encode(), v.tobytes())
            txn.put(f"meta:{k}:shape".encode(), np.array(v.shape, dtype=np.int64).tobytes())
            txn.put(f"meta:{k}:dtype".encode(), str(v.dtype).encode())

        txn.put(b"n_frames", struct.pack("<I", n_frames))
        txn.put(b"img_size", struct.pack("<HH", cfg.img_size, cfg.img_size))
        txn.put(b"channels", struct.pack("<H", c))
        txn.put(b"meta_keys", ",".join(meta.keys()).encode())
        txn.put(b"encoding", cfg.encoding.encode())

    batch_size = cfg.batch_size
    total_bytes = 0
    with h5py.File(h5_path, "r") as f:
        pixels = f[cfg.pixel_key]
        for start in tqdm(range(0, n_frames, batch_size), desc="Converting"):
            end = min(start + batch_size, n_frames)
            batch = pixels[start:end]

            with env.begin(write=True) as txn:
                for i in range(end - start):
                    encoded = encode_frame(batch[i], cfg)
                    txn.put(struct.pack("<I", start + i), encoded)
                    total_bytes += len(encoded)

    env.close()
    actual_size = sum(p.stat().st_size for p in out_path.iterdir())
    print(f"Avg frame size: {total_bytes / n_frames:.0f} bytes")
    print(f"Done. LMDB size: {actual_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
