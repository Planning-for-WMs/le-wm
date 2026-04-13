"""Closed-loop MPC evaluation of LeWM on `swm/PushT-v1`.

Mirrors `eval_antmaze_explore.py`: loads the trained LeWM checkpoint via
`swm.policy.AutoCostModel`, runs a CEM planner in a closed-loop MPC against
the real env, logs rollout videos + prediction MSE + latent projections to
wandb. Success criterion matches the official PushT setup:

    pos_diff  = ‖[agent_xy, block_xy] - [goal_agent_xy, goal_block_xy]‖  (4 dims)
    angle_diff = short-arc(block_angle - goal_block_angle)
    success    = pos_diff < pos_threshold AND angle_diff < angle_threshold
"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")

from collections import deque
from pathlib import Path

import cv2
import gymnasium as gym
import hydra
import hydra.utils
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torchvision.transforms import v2 as transforms

# Injects `swm.data.LMDBDataset` (lives in local_lmdb.py).
import local_lmdb  # noqa: F401


# ---------------------------------------------------------------------------
# Transforms & helpers
# ---------------------------------------------------------------------------

def make_image_transform(img_size: int):
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=img_size),
    ])


def fit_action_scaler(dataset) -> preprocessing.StandardScaler:
    data = dataset.get_col_data("action")
    data = data[~np.isnan(data).any(axis=1)]
    return preprocessing.StandardScaler().fit(data)


def frame_to_model_input(frame: np.ndarray, transform, device) -> torch.Tensor:
    """(H, W, 3) uint8 -> (1, 1, 3, H, W) float on device."""
    return transform(frame).unsqueeze(0).unsqueeze(0).to(device)


def pos_angle_diff(info: dict, goal_state: np.ndarray):
    """Official PushT metric: 4-d agent+block pos diff, modular block angle diff."""
    bp = info["block_pose"]
    agent_xy = np.asarray(info["pos_agent"], dtype=np.float64)
    block_xy = bp[:2].astype(np.float64)
    block_ang = float(bp[2])
    cur4 = np.concatenate([agent_xy, block_xy])
    goal4 = np.concatenate([goal_state[0:2], goal_state[2:4]])
    pos_diff = float(np.linalg.norm(cur4 - goal4))
    # Shortest-arc wrap on [-π, π].
    ang_diff = abs((block_ang - float(goal_state[4]) + np.pi) % (2 * np.pi) - np.pi)
    return pos_diff, ang_diff


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def run_rollout(
    cost_model,
    env,
    solver,
    start_state: np.ndarray,
    goal_state: np.ndarray,
    goal_frame_raw: np.ndarray,
    img_transform,
    action_scaler: preprocessing.StandardScaler,
    plan_cfg: DictConfig,
    eval_budget: int,
    pos_thresh: float,
    ang_thresh: float,
    device,
    history_size: int,
):
    env.reset()
    env.unwrapped._set_state(start_state)
    env.unwrapped._set_goal_state(goal_state)

    action_block = plan_cfg.action_block
    receding = plan_cfg.receding_horizon
    exec_per_plan = receding * action_block
    raw_action_dim = int(np.prod(env.action_space.shape))

    dummy_action = torch.zeros(1, raw_action_dim, device=device)
    goal_single = frame_to_model_input(goal_frame_raw, img_transform, device)
    goal_tensor = goal_single.expand(-1, history_size, -1, -1, -1)

    frame_history = deque(maxlen=history_size)
    cur_frame = env.render()
    # Seed history with a single frame; predictor handles T < history_size.
    frame_history.append(cur_frame.copy())

    def _pixels_tensor():
        tensors = [frame_to_model_input(f, img_transform, device) for f in frame_history]
        x = torch.cat(tensors, dim=1)          # (1, len, 3, H, W)
        # Left-pad by tiling the oldest frame so shape matches history_size.
        if x.shape[1] < history_size:
            pad = x[:, :1].expand(-1, history_size - x.shape[1], -1, -1, -1)
            x = torch.cat([pad, x], dim=1)
        return x

    frames = [cur_frame.copy()]
    pos_diffs, ang_diffs, step_success = [], [], []
    plan_actions_raw = None
    plan_pred_latents = None
    step_in_plan = 0
    predicted_emb, actual_emb = [], []

    with torch.inference_mode():
        for t in range(eval_budget):
            if plan_actions_raw is None or step_in_plan >= exec_per_plan:
                pixels_in = _pixels_tensor()
                info_dict = {"pixels": pixels_in, "goal": goal_tensor, "action": dummy_action}
                outputs = solver.solve(info_dict, init_action=None)
                actions = outputs["actions"][0].cpu().numpy()                  # (H, raw * block)
                take = actions[:receding]
                plan_actions_normed = take.reshape(receding * action_block, raw_action_dim)
                plan_actions_raw = action_scaler.inverse_transform(plan_actions_normed)
                plan_actions_raw = np.clip(
                    plan_actions_raw, env.action_space.low, env.action_space.high,
                ).astype(np.float32)

                chosen = outputs["actions"][:1, None, :, :]
                roll_info = cost_model.rollout(
                    {"pixels": pixels_in.unsqueeze(1), "action": None},
                    chosen.to(device),
                    history_size=history_size,
                )
                plan_pred_latents = roll_info["predicted_emb"][0, 0].float().cpu().numpy()
                step_in_plan = 0

            obs, _, term, trunc, info = env.step(plan_actions_raw[step_in_plan])
            cur_frame = env.render()
            frame_history.append(cur_frame.copy())
            frames.append(cur_frame.copy())

            pd, ad = pos_angle_diff(info, goal_state)
            pos_diffs.append(pd); ang_diffs.append(ad)
            step_success.append(pd < pos_thresh and ad < ang_thresh)

            # Latent trace for logging.
            plan_idx = min(step_in_plan // action_block + 1, plan_pred_latents.shape[0] - 1)
            predicted_emb.append(plan_pred_latents[plan_idx])
            enc = cost_model.encode({"pixels": frame_to_model_input(cur_frame, img_transform, device)})
            actual_emb.append(enc["emb"][0, 0].float().cpu().numpy())

            step_in_plan += 1
            if term or trunc:
                break

    return dict(
        frames=np.stack(frames),
        pos_diffs=np.array(pos_diffs),
        angle_diffs=np.array(ang_diffs),
        success=bool(np.any(step_success)) if step_success else False,
        min_pos_diff=float(np.min(pos_diffs)) if pos_diffs else float("inf"),
        min_angle_diff=float(np.min(ang_diffs)) if ang_diffs else float("inf"),
        predicted_emb=np.stack(predicted_emb) if predicted_emb else np.empty((0,)),
        actual_emb=np.stack(actual_emb) if actual_emb else np.empty((0,)),
        goal_frame=goal_frame_raw,
    )


# ---------------------------------------------------------------------------
# Visualization helpers (matching eval_antmaze_explore style)
# ---------------------------------------------------------------------------

def overlay_goal(frames: np.ndarray, goal_frame: np.ndarray) -> np.ndarray:
    out = frames.copy()
    h, w = frames.shape[1:3]
    gh, gw = h // 4, w // 4
    goal_small = cv2.resize(goal_frame, (gw, gh), interpolation=cv2.INTER_AREA)
    out[:, :gh, -gw:] = goal_small
    out[:, :gh, -gw - 1 : -gw] = 255
    out[:, gh : gh + 1, -gw:] = 255
    return out


def collect_val_latents(cost_model, dataset, img_transform, num_points: int, device):
    n_total = int(dataset.lengths.sum())
    idxs = np.random.default_rng(0).integers(0, n_total, size=min(num_points, n_total))
    latents, states = [], []
    with torch.inference_mode():
        for i in idxs:
            row = dataset.get_row_data([int(i)])
            frame_np = row["pixels"][0]
            frame_t = frame_to_model_input(frame_np, img_transform, device)
            out = cost_model.encode({"pixels": frame_t})
            latents.append(out["emb"][0, 0].float().cpu().numpy())
            if "observation" in row:
                states.append(row["observation"][0, :2].astype(np.float32))
            else:
                states.append(np.zeros(2, dtype=np.float32))
    return np.stack(latents), np.stack(states)


def log_rollout_artifacts(rollouts, val_latents, val_states, run_dir, wandb_run, fps=10):
    import wandb

    # Videos (env frames with goal overlaid in the corner).
    for i, r in enumerate(rollouts):
        if r["frames"].shape[0] == 0:
            continue
        overlaid = overlay_goal(r["frames"], r["goal_frame"])
        T, H, W, _ = overlaid.shape
        mp4 = run_dir / f"rollout_{i:02d}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(mp4), fourcc, fps, (W, H))
        for f in overlaid:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()
        wandb_run.log({f"rollout/video_{i:02d}": wandb.Video(str(mp4))})

    # Prediction MSE curve.
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, r in enumerate(rollouts):
        if r["actual_emb"].size == 0:
            continue
        mse = ((r["predicted_emb"] - r["actual_emb"]) ** 2).mean(axis=-1)
        ax.plot(mse, alpha=0.7, label=f"rollout {i}")
    ax.set_xlabel("env step"); ax.set_ylabel("MSE(pred, actual) in latent")
    ax.set_yscale("log"); ax.legend(fontsize=7)
    fig.tight_layout()
    wandb_run.log({"rollout/latent_mse": wandb.Image(fig)}); plt.close(fig)

    # pos_diff / angle_diff curves.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i, r in enumerate(rollouts):
        axes[0].plot(r["pos_diffs"], alpha=0.6, label=f"rollout {i}")
        axes[1].plot(r["angle_diffs"], alpha=0.6, label=f"rollout {i}")
    axes[0].axhline(20, color="k", linestyle="--", label="thresh=20")
    axes[1].axhline(np.pi / 9, color="k", linestyle="--", label="thresh=π/9")
    axes[0].set_title("pos_diff vs env step"); axes[1].set_title("angle_diff vs env step")
    for ax in axes: ax.legend(fontsize=7)
    fig.tight_layout()
    wandb_run.log({"rollout/distance_to_goal": wandb.Image(fig)}); plt.close(fig)

    # PCA + t-SNE of val latents with rollout trajectories overlaid.
    rollout_latents, rollout_ids = [], []
    for i, r in enumerate(rollouts):
        if r["actual_emb"].size:
            rollout_latents.append(r["actual_emb"])
            rollout_ids.extend([i] * r["actual_emb"].shape[0])
    rollout_latents = (np.concatenate(rollout_latents, axis=0) if rollout_latents
                       else np.empty((0, val_latents.shape[1])))
    rollout_ids = np.asarray(rollout_ids)

    combined = (np.concatenate([val_latents, rollout_latents], axis=0)
                if rollout_latents.size else val_latents)

    pca = PCA(n_components=2).fit(val_latents)
    val_pca = pca.transform(val_latents)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(val_pca[:, 0], val_pca[:, 1], c=val_states[:, 0], cmap="viridis",
               s=6, alpha=0.5, label="val latents")
    if rollout_latents.size:
        rol_pca = pca.transform(rollout_latents)
        for i in range(len(rollouts)):
            mask = rollout_ids == i
            if mask.any():
                ax.plot(rol_pca[mask, 0], rol_pca[mask, 1], "-o", ms=3, label=f"rollout {i}")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend(fontsize=7)
    ax.set_title("PCA of encoder latents + rollouts")
    fig.tight_layout()
    wandb_run.log({"latent/pca_with_rollouts": wandb.Image(fig)}); plt.close(fig)

    perp = min(30, max(5, combined.shape[0] // 4))
    tsne = TSNE(n_components=2, perplexity=perp, init="pca",
                random_state=0, learning_rate="auto").fit_transform(combined)
    val_tsne = tsne[: val_latents.shape[0]]
    rol_tsne = tsne[val_latents.shape[0]:]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(val_tsne[:, 0], val_tsne[:, 1], c=val_states[:, 0], cmap="viridis",
               s=6, alpha=0.5, label="val latents")
    if rol_tsne.size:
        for i in range(len(rollouts)):
            mask = rollout_ids == i
            if mask.any():
                ax.plot(rol_tsne[mask, 0], rol_tsne[mask, 1], "-o", ms=3, label=f"rollout {i}")
    ax.set_title("t-SNE of encoder latents + rollouts"); ax.legend(fontsize=7)
    fig.tight_layout()
    wandb_run.log({"latent/tsne_with_rollouts": wandb.Image(fig)}); plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="./config/eval", config_name="lewm_pusht")
def run(cfg: DictConfig):
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache_dir = Path(cfg.cache_dir)
    raw_dataset = swm.data.LMDBDataset(
        name=cfg.dataset.name,
        cache_dir=cache_dir,
        keys_to_load=list(cfg.dataset.keys_to_load),
        frameskip=1,
        num_steps=1,
        transform=None,
    )
    action_scaler = fit_action_scaler(raw_dataset)
    img_transform = make_image_transform(cfg.env.img_size)

    print(f"Loading model: {cfg.policy}")
    cost_model = swm.policy.AutoCostModel(cfg.policy, cache_dir=cache_dir).to(device).eval()
    cost_model.requires_grad_(False)

    env = gym.make(
        cfg.env.name,
        max_episode_steps=cfg.env.max_episode_steps,
        render_mode="rgb_array",
    )
    env.reset()

    solver = hydra.utils.instantiate(cfg.solver, model=cost_model, device=str(device))
    solver.configure(
        action_space=env.action_space, n_envs=1,
        config=swm.PlanConfig(**cfg.plan_config),
    )
    solver._action_dim = int(np.prod(env.action_space.shape))

    # Pick random episodes; goal is at (start_step + goal_offset_steps) within
    # the same episode, so episodes must be at least `goal_offset_steps + 1`.
    rng = np.random.default_rng(cfg.eval.seed)
    ep_lens, ep_offs = raw_dataset.lengths, raw_dataset.offsets
    valid = np.where(ep_lens > cfg.eval.goal_offset_steps)[0]
    chosen = rng.choice(valid, size=min(cfg.eval.num_rollouts, len(valid)), replace=False)

    run_dir = Path(swm.data.utils.get_cache_dir(), "eval_pusht")
    run_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = None
    if cfg.wandb.enabled:
        import wandb
        wandb_run = wandb.init(
            entity=cfg.wandb.entity, project=cfg.wandb.project, name=cfg.wandb.name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    rollouts = []
    for i, ep in enumerate(chosen):
        o = int(ep_offs[ep])
        start_row = raw_dataset.get_row_data([o])
        goal_row = raw_dataset.get_row_data([o + int(cfg.eval.goal_offset_steps)])
        start_state = start_row["state"][0]
        goal_state = goal_row["state"][0]
        goal_frame = goal_row["pixels"][0]
        print(f"\n=== rollout {i + 1}/{len(chosen)} (ep {ep}) ===")
        r = run_rollout(
            cost_model=cost_model, env=env, solver=solver,
            start_state=start_state, goal_state=goal_state, goal_frame_raw=goal_frame,
            img_transform=img_transform, action_scaler=action_scaler,
            plan_cfg=cfg.plan_config, eval_budget=cfg.eval.eval_budget,
            pos_thresh=cfg.eval.pos_threshold, ang_thresh=cfg.eval.angle_threshold,
            device=device, history_size=cfg.history_size,
        )
        rollouts.append(r)
        print(f"  min_pos={r['min_pos_diff']:.1f} (thresh={cfg.eval.pos_threshold})  "
              f"min_ang={r['min_angle_diff']:.3f} (thresh={cfg.eval.angle_threshold:.3f})  "
              f"success={r['success']}")

    success_rate = float(np.mean([r["success"] for r in rollouts]))
    mean_min_pos = float(np.mean([r["min_pos_diff"] for r in rollouts]))
    mean_min_ang = float(np.mean([r["min_angle_diff"] for r in rollouts]))
    print(f"\n=== SUMMARY ===")
    print(f"success_rate:       {success_rate:.3f}  ({sum(r['success'] for r in rollouts)}/{len(rollouts)})")
    print(f"mean min_pos_diff:  {mean_min_pos:.2f}")
    print(f"mean min_angle:     {mean_min_ang:.3f}")

    if wandb_run is not None:
        val_latents, val_states = collect_val_latents(
            cost_model, raw_dataset, img_transform, cfg.latent_viz.num_points, device,
        )
        log_rollout_artifacts(rollouts, val_latents, val_states, run_dir, wandb_run)
        wandb_run.log({"summary/success_rate": success_rate,
                       "summary/mean_min_pos_diff": mean_min_pos,
                       "summary/mean_min_angle_diff": mean_min_ang})
        wandb_run.finish()

    out_path = Path(cfg.output.filename)
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a") as f:
        f.write("\n==== CONFIG ====\n"); f.write(OmegaConf.to_yaml(cfg))
        f.write("\n==== RESULTS ====\n")
        f.write(f"success_rate: {success_rate:.3f}\n")
        f.write(f"mean_min_pos_diff: {mean_min_pos:.2f}\n")
        f.write(f"mean_min_angle_diff: {mean_min_ang:.3f}\n")
    print(f"Wrote summary to {out_path}")


if __name__ == "__main__":
    run()
