"""Closed-loop CEM evaluation of DINO-WM on PushT with ToMe-accelerated planning.

Uses the recorded `pusht_50steps` dataset for start/goal pairs. The model is
loaded from a downloaded DINO-WM checkpoint, the predictor's rollout is
monkey-patched with `tome_rollout`, and CEM optimizes 5-action plan blocks
(action_block=5 to match DINO-WM's frameskip).
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import sys
import time
import types
from collections import deque
from pathlib import Path

import cv2
import gymnasium as gym
import hydra
import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf

from hierarchical.dino_wm_wrapper import load_dino_wm
from hierarchical.tome_predictor import tome_rollout


# Action / proprio normalization stats from the original DINO-WM PushT dataset.
ACTION_MEAN = np.array([-0.0087, 0.0068], dtype=np.float32)
ACTION_STD = np.array([0.2019, 0.2002], dtype=np.float32)
PROPRIO_MEAN = np.array([236.6155, 264.5674, -2.93032027, 2.54307914], dtype=np.float32)
PROPRIO_STD = np.array([101.1202, 87.0112, 74.84556075, 74.14009094], dtype=np.float32)


def img_to_tensor(frame_uint8: np.ndarray, device) -> torch.Tensor:
    """(H, W, 3) uint8 -> (3, H, W) float in [-1, 1] (DINO-WM normalization)."""
    t = torch.from_numpy(frame_uint8).permute(2, 0, 1).float() / 255.0
    t = (t - 0.5) / 0.5
    return t.to(device)


def normalize_proprio(p: np.ndarray) -> np.ndarray:
    return ((p - PROPRIO_MEAN) / PROPRIO_STD).astype(np.float32)


def normalize_action(a: np.ndarray) -> np.ndarray:
    return ((a - ACTION_MEAN) / ACTION_STD).astype(np.float32)


def denormalize_action(a: np.ndarray) -> np.ndarray:
    return (a * ACTION_STD + ACTION_MEAN).astype(np.float32)


def save_video(frames: np.ndarray, path: str, fps: int = 10):
    T, H, W, C = frames.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (W, H))
    for t in range(T):
        writer.write(cv2.cvtColor(frames[t], cv2.COLOR_RGB2BGR))
    writer.release()


def overlay_goal(frames: np.ndarray, goal_frame: np.ndarray) -> np.ndarray:
    out = frames.copy()
    h, w = frames.shape[1:3]
    gh, gw = h // 4, w // 4
    goal_small = cv2.resize(goal_frame, (gw, gh), interpolation=cv2.INTER_AREA)
    out[:, :gh, -gw:] = goal_small
    out[:, :gh, -gw - 1 : -gw] = 255
    out[:, gh : gh + 1, -gw:] = 255
    return out


def run_episode(
    model,
    env,
    solver,
    start_state: np.ndarray,
    start_proprio: np.ndarray,
    goal_state: np.ndarray,
    goal_frame_uint8: np.ndarray,
    cfg,
    device,
    episode_idx: int,
):
    """Closed-loop CEM rollout for a single episode."""
    T_hist = model.num_hist
    action_block = cfg.plan_config.action_block
    receding = cfg.plan_config.receding_horizon

    # Reset env, then jump to the chosen start state. Render after.
    env.reset()
    env.unwrapped._set_state(start_state)
    env.unwrapped._set_goal_state(goal_state)
    cur_frame = env.render()
    cur_proprio = start_proprio.astype(np.float32)

    # Encode the goal once.
    goal_t = img_to_tensor(goal_frame_uint8, device).unsqueeze(0)  # (1, 3, H, W)
    model.set_goal(goal_t)

    # History buffers (one entry per "plan step" = action_block env steps).
    pixel_hist = deque(maxlen=T_hist)
    proprio_hist = deque(maxlen=T_hist)
    action_hist = deque(maxlen=T_hist)

    init_pix = img_to_tensor(cur_frame, device)
    init_pro = torch.from_numpy(normalize_proprio(cur_proprio)).to(device)
    for _ in range(T_hist):
        pixel_hist.append(init_pix)
        proprio_hist.append(init_pro)
        action_hist.append(torch.zeros(2 * action_block, device=device))

    rendered_frames = [cur_frame.copy()]
    block_dists = []
    plan_actions = None  # (horizon, raw_action_dim * action_block)
    plan_step_idx = 0
    raw_action_buffer = []   # collects raw actions until we have one full plan-step

    success = False
    final_dist = float("inf")

    with torch.inference_mode():
        for env_step in range(cfg.eval.eval_budget):
            need_replan = (plan_actions is None) or (plan_step_idx >= receding)
            if need_replan:
                pixels_in = torch.stack(list(pixel_hist)).unsqueeze(0)      # (1, T_hist, 3, H, W)
                proprio_in = torch.stack(list(proprio_hist)).unsqueeze(0)   # (1, T_hist, 4)
                action_in = torch.stack(list(action_hist)).unsqueeze(0)     # (1, T_hist, 10)

                info_dict = {
                    "pixels": pixels_in,
                    "proprio": proprio_in,
                    "action": action_in,
                }
                t_replan = time.time()
                outputs = solver.solve(info_dict)
                if env_step == 0:
                    print(f"  episode {episode_idx}: first replan {time.time() - t_replan:.2f}s")
                plan_actions = outputs["actions"][0].cpu().numpy()  # (horizon, 10)
                plan_step_idx = 0

            # Execute one env step from the current plan-step.
            sub_idx = env_step % action_block
            current_block = plan_actions[plan_step_idx]                 # (10,)
            normed_raw = current_block.reshape(action_block, 2)[sub_idx]  # (2,) — normalized
            raw = denormalize_action(normed_raw)
            raw = np.clip(raw, env.action_space.low, env.action_space.high).astype(np.float32)

            obs, rew, term, trunc, step_info = env.step(raw)
            cur_frame = env.render()
            rendered_frames.append(cur_frame.copy())

            # Track block-pose distance to goal pose
            bp = step_info.get("block_pose")
            gp = step_info.get("goal_pose")
            if bp is not None and gp is not None:
                d = float(np.linalg.norm(bp[:2] - gp[:2]))
                block_dists.append(d)
                final_dist = d

            raw_action_buffer.append(normed_raw.astype(np.float32))

            # When a full plan-step has been executed, advance history.
            if (env_step + 1) % action_block == 0:
                pixel_hist.append(img_to_tensor(cur_frame, device))
                cur_proprio = obs["proprio"].astype(np.float32)
                proprio_hist.append(torch.from_numpy(normalize_proprio(cur_proprio)).to(device))
                action_hist.append(
                    torch.from_numpy(np.concatenate(raw_action_buffer)).to(device)
                )
                raw_action_buffer = []
                plan_step_idx += 1

            if term or trunc:
                break

    # Success: block within ~5 pixels of goal pose at any point.
    success = bool(min(block_dists) < 10.0) if block_dists else False

    return dict(
        frames=np.stack(rendered_frames),
        block_dists=np.array(block_dists),
        final_dist=final_dist,
        success=success,
        goal_frame=goal_frame_uint8,
    )


@hydra.main(version_base=None, config_path="./config/eval", config_name="dino")
def run(cfg: DictConfig):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Dataset for start/goal pairs
    dataset = swm.data.HDF5Dataset(
        name=cfg.dataset.name,
        cache_dir=Path(cfg.dataset.cache_dir),
        keys_to_cache=["action", "proprio", "state", "goal_state"],
        num_steps=1,
        frameskip=1,
        transform=None,
    )
    print(f"Dataset: {len(dataset.lengths)} episodes, lengths={dataset.lengths[:5]}...")

    # --- Model
    print(f"Loading checkpoint: {cfg.checkpoint}")
    model = load_dino_wm(cfg.checkpoint, device=str(device))
    print(
        f"Loaded DINO-WM (raw_action_dim={model.raw_action_dim}, "
        f"raw_proprio_dim={model.raw_proprio_dim})"
    )

    # --- Patch rollout with ToMe
    tome_r = cfg.tome.r

    def patched_rollout(self_, info_, actions_):
        return tome_rollout(self_, info_, actions_, r=tome_r)

    model.rollout = types.MethodType(patched_rollout, model)
    print(f"Patched rollout with ToMe (r={tome_r})")

    # --- Env (gym.make works after `import stable_worldmodel`)
    env = gym.make(
        cfg.env.name,
        max_episode_steps=cfg.env.max_episode_steps,
        render_mode="rgb_array",
    )

    # --- CEM solver. PushT raw action_dim=2; per-step action_dim becomes 2*5=10.
    solver = swm.solver.CEMSolver(
        model=model,
        batch_size=cfg.solver.batch_size,
        num_samples=cfg.solver.num_samples,
        var_scale=cfg.solver.var_scale,
        n_steps=cfg.solver.n_steps,
        topk=cfg.solver.topk,
        device=str(device),
    )
    plan_cfg = swm.PlanConfig(**cfg.plan_config)
    # CEMSolver._action_dim is np.prod(action_space.shape[1:]); for PushT
    # (vector env shape (1,2)) shape[1:]=(2,) — that's already correct.
    solver.configure(action_space=env.action_space, n_envs=1, config=plan_cfg)
    solver._action_dim = int(np.prod(env.action_space.shape))   # 2 (raw)
    print(f"CEM action_dim per plan-step = {solver.action_dim} (= 2 * action_block)")

    # --- Pick episodes
    rng = np.random.default_rng(cfg.seed)
    n_eps = len(dataset.lengths)
    picked = rng.choice(n_eps, size=min(cfg.eval.num_eval, n_eps), replace=False)

    video_dir = Path(cfg.output.video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, ep_idx in enumerate(picked):
        offset = int(dataset.offsets[ep_idx])
        ep_len = int(dataset.lengths[ep_idx])
        goal_frame_in_ep = min(cfg.eval.goal_offset_steps, ep_len - 1)

        start_row = dataset.get_row_data([offset])
        goal_row = dataset.get_row_data([offset + goal_frame_in_ep])
        start_state = start_row["state"][0]
        goal_state = goal_row["state"][0]
        start_proprio = start_row["proprio"][0]
        # Goal frame: use the rendered pixels at the chosen goal step.
        goal_frame_uint8 = goal_row["pixels"][0]

        print(f"\n=== Episode {i + 1}/{len(picked)} (ep_idx={ep_idx}) ===")
        result = run_episode(
            model=model,
            env=env,
            solver=solver,
            start_state=start_state,
            start_proprio=start_proprio,
            goal_state=goal_state,
            goal_frame_uint8=goal_frame_uint8,
            cfg=cfg,
            device=device,
            episode_idx=int(ep_idx),
        )
        results.append(result)
        print(
            f"  steps={len(result['frames']) - 1}, "
            f"min_block_dist={min(result['block_dists']):.1f} "
            f"(goal_thresh=10), success={result['success']}"
        )

        if cfg.eval.save_videos:
            overlaid = overlay_goal(result["frames"], result["goal_frame"])
            path = video_dir / f"eval_ep{i}.mp4"
            save_video(overlaid, str(path))
            print(f"  wrote {path}")

    # --- Summary
    success_rate = float(np.mean([r["success"] for r in results]))
    mean_min_dist = float(np.mean([r["block_dists"].min() for r in results if len(r["block_dists"])]))
    print(f"\n=== SUMMARY ===")
    print(f"success_rate: {success_rate:.3f} ({sum(r['success'] for r in results)}/{len(results)})")
    print(f"mean_min_block_dist: {mean_min_dist:.2f}")
    print(f"tome_r: {tome_r}")

    out_path = Path(cfg.output.filename)
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a") as f:
        f.write("\n==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n==== RESULTS ====\n")
        f.write(f"num_episodes: {len(results)}\n")
        f.write(f"success_rate: {success_rate:.3f}\n")
        f.write(f"mean_min_block_dist: {mean_min_dist:.2f}\n")
        f.write(f"tome_r: {tome_r}\n")
    print(f"Wrote summary to {out_path}")


if __name__ == "__main__":
    run()
