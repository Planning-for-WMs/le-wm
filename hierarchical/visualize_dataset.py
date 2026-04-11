"""Visualize a few trajectories from the PushT dataset as mp4 files."""

import numpy as np
import stable_worldmodel as swm
from pathlib import Path

import cv2


def save_trajectory_mp4(frames: np.ndarray, path: str, fps: int = 10):
    """Save (T, H, W, 3) uint8 frames as mp4."""
    T, H, W, C = frames.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (W, H))
    for t in range(T):
        writer.write(cv2.cvtColor(frames[t], cv2.COLOR_RGB2BGR))
    writer.release()


def main():
    dataset = swm.data.HDF5Dataset(
        name="pusht_50steps",
        keys_to_cache=["action"],
        num_steps=50,
        frameskip=1,
    )
    print(f"Dataset: {len(dataset)} windows, columns: {dataset.column_names}")

    out_dir = Path(__file__).parent / "viz"
    out_dir.mkdir(exist_ok=True)

    n_trajs = min(3, len(dataset))
    for i in range(n_trajs):
        sample = dataset[i]
        pixels = sample["pixels"]  # (T, C, H, W) or (T, H, W, C) depending on format
        if isinstance(pixels, np.ndarray):
            if pixels.ndim == 4 and pixels.shape[1] == 3:
                pixels = pixels.transpose(0, 2, 3, 1)  # TCHW -> THWC
        else:
            import torch
            if pixels.dim() == 4 and pixels.shape[1] == 3:
                pixels = pixels.permute(0, 2, 3, 1)
            pixels = pixels.numpy()

        # Ensure uint8
        if pixels.dtype != np.uint8:
            if pixels.max() <= 1.0:
                pixels = (pixels * 255).clip(0, 255).astype(np.uint8)
            else:
                pixels = pixels.clip(0, 255).astype(np.uint8)

        path = out_dir / f"traj_{i}.mp4"
        save_trajectory_mp4(pixels, str(path))
        print(f"Saved {path} — shape {pixels.shape}")


if __name__ == "__main__":
    main()
