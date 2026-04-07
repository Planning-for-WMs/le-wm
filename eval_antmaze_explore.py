"""Closed-loop MPC evaluation of LeWM on og-bench antmaze-medium.

For each of ``eval.num_rollouts`` randomly-sampled start+goal pairs from the
LMDB, run the world model as a CEM planner for ``eval.traj_steps`` env steps,
replanning every ``plan_config.receding_horizon * plan_config.action_block``
env steps. Log to wandb:

    - rollout videos (RGB with goal overlaid in the corner)
    - prediction MSE vs rollout step
    - t-SNE of val-set latents with rollout trajectory overlaid
    - PCA(2) of val-set latents with rollout trajectory overlaid

Also write ``antmaze_results.txt`` next to the checkpoint with success rate
and mean prediction MSE for quick comparison across runs.
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import io
from pathlib import Path

import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import ogbench
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torchvision.transforms import v2 as transforms

import stable_pretraining as spt


# ---------------------------------------------------------------------------
# Transforms & normalization helpers
# ---------------------------------------------------------------------------

def make_image_transform(img_size: int):
    """ImageNet-normalised resize + to-tensor transform (matches train pipeline)."""
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=img_size),
        ]
    )


def fit_action_scaler(dataset) -> preprocessing.StandardScaler:
    data = dataset.get_col_data("action")
    data = data[~np.isnan(data).any(axis=1)]
    scaler = preprocessing.StandardScaler().fit(data)
    return scaler


def frame_to_model_input(frame: np.ndarray, transform, device) -> torch.Tensor:
    """(H, W, 3) uint8 -> (1, 1, 3, H, W) float on device."""
    x = transform(frame)  # (3, H, W)
    return x.unsqueeze(0).unsqueeze(0).to(device)


# ---------------------------------------------------------------------------
# Rollout loop
# ---------------------------------------------------------------------------

def run_rollout(
    cost_model,
    env,
    solver,
    start_state: np.ndarray,
    goal_frame_raw: np.ndarray,
    goal_state: np.ndarray,
    img_transform,
    action_scaler: preprocessing.StandardScaler,
    action_block: int,
    receding_horizon: int,
    traj_steps: int,
    device,
    history_size: int = 3,
):
    """Single closed-loop MPC rollout.

    Returns a dict of per-step arrays suitable for plotting:
        frames          : (traj_steps, H, W, 3) uint8
        xys             : (traj_steps, 2) agent xy position
        predicted_emb   : (traj_steps, D) model's most-recent prediction of this step's latent
        actual_emb      : (traj_steps, D) encoder's latent for the observed frame at this step
        success_step    : int or -1
        goal_xy         : (2,)
    """
    nq = env.unwrapped.model.nq
    nv = env.unwrapped.model.nv

    # Reset and seed env to the requested start state.
    env.reset()
    qpos = start_state[:nq].astype(np.float64)
    qvel = start_state[nq : nq + nv].astype(np.float64)
    env.unwrapped.set_state(qpos, qvel)

    frames = []
    xys = []
    predicted_emb = []  # aligned with executed env steps
    actual_emb = []
    exec_per_plan = receding_horizon * action_block

    cur_frame = env.render()
    cur_xy = env.unwrapped.data.qpos[:2].copy()

    success_step = -1
    goal_xy = goal_state[:2].copy()

    plan_actions_raw = None
    plan_pred_latents = None
    step_in_plan = 0
    replan_dists = []  # (env_step, distance_to_goal) at each replan

    raw_action_dim = env.action_space.shape[0]
    # Dummy action for JEPA.get_cost which pops "action" from the goal dict.
    dummy_action = torch.zeros(1, raw_action_dim, device=device)

    # History buffer: the predictor was trained on ``history_size`` context
    # frames. We maintain a sliding window of rendered frames and tile the
    # goal to the same length so the CEM planner and the rollout latent-trace
    # receive the same context shape the predictor expects.
    from collections import deque
    frame_history = deque(maxlen=history_size)
    # Seed the buffer with copies of the initial frame (standard practice
    # when no real history is available yet).
    for _ in range(history_size):
        frame_history.append(cur_frame.copy())

    def _pixels_tensor():
        """Stack the frame history into (1, history_size, 3, H, W)."""
        tensors = [frame_to_model_input(f, img_transform, device) for f in frame_history]
        return torch.cat(tensors, dim=1)  # each is (1,1,3,H,W) → cat on dim 1

    # Goal: tile to (1, history_size, 3, H, W) so shapes match through
    # get_cost → encode.
    goal_single = frame_to_model_input(goal_frame_raw, img_transform, device)  # (1,1,3,H,W)
    goal_tensor = goal_single.expand(-1, history_size, -1, -1, -1)  # (1, HS, 3, H, W)

    with torch.inference_mode():
        for t in range(traj_steps):
            # Replan?
            if plan_actions_raw is None or step_in_plan >= exec_per_plan:
                replan_dists.append((t, float(np.linalg.norm(cur_xy - goal_xy))))
                pixels_in = _pixels_tensor()  # (1, history_size, 3, H, W)
                info_dict = {
                    "pixels": pixels_in,
                    "goal": goal_tensor,
                    "action": dummy_action,
                }
                outputs = solver.solve(info_dict, init_action=None)
                actions = outputs["actions"][0].cpu().numpy()  # (horizon, raw*action_block)
                take = actions[:receding_horizon]
                plan_actions_normed = take.reshape(
                    receding_horizon * action_block, raw_action_dim
                )
                plan_actions_raw = action_scaler.inverse_transform(plan_actions_normed)
                plan_actions_raw = np.clip(
                    plan_actions_raw,
                    env.action_space.low,
                    env.action_space.high,
                )

                # Record the model's predicted latents for this plan.
                chosen = outputs["actions"][:1, None, :, :]  # (1, 1, horizon, dim)
                roll_info = cost_model.rollout(
                    {
                        "pixels": pixels_in.unsqueeze(1),  # (1, 1, HS, 3, H, W)
                        "action": None,
                    },
                    chosen.to(device),
                    history_size=history_size,
                )
                # predicted_emb shape: (B=1, S=1, T, D)
                pred_latents = roll_info["predicted_emb"][0, 0].float().cpu().numpy()
                # pred_latents[0] is the encoded current frame, pred_latents[k] is the model's
                # prediction of the frame after k env plan-steps forward.
                plan_pred_latents = pred_latents
                step_in_plan = 0

            # Execute one env step.
            action = plan_actions_raw[step_in_plan]
            _, _, terminated, truncated, _ = env.step(action)
            cur_frame = env.render()
            cur_xy = env.unwrapped.data.qpos[:2].copy()
            frame_history.append(cur_frame.copy())

            # Log: predicted latent for this env step (from the current plan).
            # Plan step index = step_in_plan // action_block + 1 (since index 0 is the init frame).
            plan_idx = step_in_plan // action_block + 1
            plan_idx = min(plan_idx, plan_pred_latents.shape[0] - 1)
            pred_lat = plan_pred_latents[plan_idx]

            # Actual latent for this frame.
            frame_tensor = frame_to_model_input(cur_frame, img_transform, device)
            enc = cost_model.encode({"pixels": frame_tensor})
            act_lat = enc["emb"][0, 0].float().cpu().numpy()

            frames.append(cur_frame.copy())
            xys.append(cur_xy.copy())
            predicted_emb.append(pred_lat)
            actual_emb.append(act_lat)

            step_in_plan += 1

            # Record first step where the agent enters the 1.0-radius goal
            # disk, but continue rolling out so rollout videos, MSE curves,
            # and latent trajectories cover the full traj_steps horizon.
            if np.linalg.norm(cur_xy - goal_xy) < 1.0 and success_step == -1:
                success_step = t

            if terminated or truncated:
                break

    return dict(
        frames=np.stack(frames) if frames else np.empty((0, 224, 224, 3), dtype=np.uint8),
        xys=np.stack(xys) if xys else np.empty((0, 2)),
        predicted_emb=np.stack(predicted_emb) if predicted_emb else np.empty((0,)),
        actual_emb=np.stack(actual_emb) if actual_emb else np.empty((0,)),
        replan_dists=replan_dists,  # list of (env_step, dist_to_goal) at each CEM replan
        success_step=success_step,
        goal_xy=goal_xy,
        goal_frame=goal_frame_raw,
    )


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def overlay_goal(frames: np.ndarray, goal_frame: np.ndarray) -> np.ndarray:
    """Paste a 1/4-size goal frame into the top-right corner of each frame."""
    out = frames.copy()
    h, w = frames.shape[1:3]
    gh, gw = h // 4, w // 4
    import cv2

    goal_small = cv2.resize(goal_frame, (gw, gh), interpolation=cv2.INTER_AREA)
    border = 1
    out[:, :gh, -gw:] = goal_small
    out[:, :gh, -gw - border : -gw] = 255
    out[:, gh : gh + border, -gw:] = 255
    return out


def collect_val_latents(cost_model, dataset, img_transform, num_points: int, device):
    """Encode a random subset of dataset frames and return (latents, states).

    Always reads **raw uint8 pixels** via ``dataset.get_row_data`` (bypasses
    any training-time transform the dataset object may carry) so that
    ``frame_to_model_input`` can apply ImageNet normalization exactly once.
    """
    n_total = int(dataset.lengths.sum())
    idxs = np.random.default_rng(0).integers(0, n_total, size=min(num_points, n_total))
    latents, states = [], []
    with torch.inference_mode():
        for i in idxs:
            row = dataset.get_row_data([int(i)])
            # get_row_data returns raw LMDB pixels: (1, H, W, 3) uint8
            frame_np = row["pixels"][0]  # (H, W, 3) uint8
            frame_t = frame_to_model_input(frame_np, img_transform, device)
            out = cost_model.encode({"pixels": frame_t})
            latents.append(out["emb"][0, 0].float().cpu().numpy())
            if "observation" in row:
                states.append(row["observation"][0, :2].astype(np.float32))
            else:
                states.append(np.zeros(2, dtype=np.float32))
    return np.stack(latents), np.stack(states)


def log_rollout_artifacts(rollouts, val_latents, val_states, run_dir, wandb_run):
    import wandb

    # --- 1. Videos
    for i, r in enumerate(rollouts):
        if r["frames"].shape[0] == 0:
            continue
        overlaid = overlay_goal(r["frames"], r["goal_frame"])
        video_chw = overlaid.transpose(0, 3, 1, 2)  # (T, 3, H, W)
        wandb_run.log({f"rollout/video_{i}": wandb.Video(video_chw, fps=20, format="mp4")})

    # --- 2. MSE-vs-step line chart, averaged across rollouts + per-rollout
    max_len = max(r["predicted_emb"].shape[0] for r in rollouts) if rollouts else 0
    if max_len > 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        all_mse = np.full((len(rollouts), max_len), np.nan)
        for i, r in enumerate(rollouts):
            if r["predicted_emb"].shape[0] == 0:
                continue
            mse = ((r["predicted_emb"] - r["actual_emb"]) ** 2).mean(axis=1)
            all_mse[i, : mse.shape[0]] = mse
            ax.plot(mse, alpha=0.5, label=f"rollout {i}")
        mean_mse = np.nanmean(all_mse, axis=0)
        ax.plot(mean_mse, color="black", linewidth=2, label="mean")
        ax.set_xlabel("rollout step")
        ax.set_ylabel("MSE(predicted_emb, actual_emb)")
        ax.set_title("Per-step prediction error over closed-loop rollout")
        ax.legend()
        fig.tight_layout()
        wandb_run.log({"rollout/mse_vs_step": wandb.Image(fig)})
        plt.close(fig)

    # --- 3. Distance-to-goal at each CEM replan step.
    has_replan_data = any(len(r.get("replan_dists", [])) > 0 for r in rollouts)
    if has_replan_data:
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, r in enumerate(rollouts):
            dists = r.get("replan_dists", [])
            if not dists:
                continue
            steps, ds = zip(*dists)
            ax.plot(steps, ds, "-o", markersize=3, alpha=0.7, label=f"rollout {i}")
        ax.set_xlabel("env step (replan point)")
        ax.set_ylabel("distance to goal (xy)")
        ax.set_title("Distance to goal at each CEM replan")
        ax.legend()
        fig.tight_layout()
        wandb_run.log({"rollout/dist_to_goal": wandb.Image(fig)})
        plt.close(fig)

    # --- 4 & 5. t-SNE + PCA of val latents with rollout latents overlaid.
    rollout_latents = np.concatenate(
        [r["actual_emb"] for r in rollouts if r["actual_emb"].shape[0] > 0], axis=0
    ) if rollouts else np.empty((0,))
    rollout_ids = np.concatenate(
        [np.full(r["actual_emb"].shape[0], i) for i, r in enumerate(rollouts) if r["actual_emb"].shape[0] > 0]
    ) if rollouts else np.empty((0,))

    if val_latents.shape[0] > 0:
        combined = (
            np.concatenate([val_latents, rollout_latents], axis=0)
            if rollout_latents.size
            else val_latents
        )

        # PCA: fit on val, project both.
        pca = PCA(n_components=2).fit(val_latents)
        val_pca = pca.transform(val_latents)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(val_pca[:, 0], val_pca[:, 1], c=val_states[:, 0], cmap="viridis",
                   s=6, alpha=0.5, label="val latents")
        if rollout_latents.size:
            rol_pca = pca.transform(rollout_latents)
            for i, r in enumerate(rollouts):
                n_steps = r["actual_emb"].shape[0]
                if n_steps == 0:
                    continue
                mask = rollout_ids == i
                pts = rol_pca[mask]
                ax.plot(pts[:, 0], pts[:, 1], "-o", markersize=3, label=f"rollout {i}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("PCA of encoder latents + closed-loop rollout trajectories")
        ax.legend()
        fig.tight_layout()
        wandb_run.log({"latent/pca_with_rollouts": wandb.Image(fig)})
        plt.close(fig)

        # t-SNE: fit on the combined set so rollout points share the embedding.
        perplexity = min(30, max(5, combined.shape[0] // 4))
        tsne = TSNE(n_components=2, perplexity=perplexity, init="pca",
                    random_state=0, learning_rate="auto")
        combined_tsne = tsne.fit_transform(combined)
        val_tsne = combined_tsne[: val_latents.shape[0]]
        rol_tsne = combined_tsne[val_latents.shape[0] :]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(val_tsne[:, 0], val_tsne[:, 1], c=val_states[:, 0], cmap="viridis",
                   s=6, alpha=0.5, label="val latents")
        if rollout_latents.size:
            for i in range(len(rollouts)):
                mask = rollout_ids == i
                if not mask.any():
                    continue
                pts = rol_tsne[mask]
                ax.plot(pts[:, 0], pts[:, 1], "-o", markersize=3, label=f"rollout {i}")
        ax.set_title("t-SNE of encoder latents + closed-loop rollout trajectories")
        ax.legend()
        fig.tight_layout()
        wandb_run.log({"latent/tsne_with_rollouts": wandb.Image(fig)})
        plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="./config/eval", config_name="antmaze")
def run(cfg: DictConfig):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Dataset (for start/goal sampling + action scaler stats + val latents)
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

    # --- Model
    print(f"Loading model: {cfg.policy}")
    cost_model = swm.policy.AutoCostModel(cfg.policy, cache_dir=cache_dir).to(device).eval()
    cost_model.requires_grad_(False)

    # --- Env. ogbench's default maze camera (no camera_name passed) is the
    # top-down view that frames the whole maze. Leave camera_name unset unless
    # the config explicitly overrides it.
    env_kwargs = dict(
        render_mode="rgb_array",
        width=cfg.env.img_size,
        height=cfg.env.img_size,
    )
    if cfg.env.get("camera_name") is not None:
        env_kwargs["camera_name"] = cfg.env.camera_name
    env = ogbench.make_env_and_datasets(
        cfg.env.dataset_name,
        env_only=True,
        **env_kwargs,
    )
    env.reset()

    # --- Solver (CEM)
    import hydra.utils

    solver = hydra.utils.instantiate(cfg.solver, model=cost_model, device=str(device))
    solver.configure(
        action_space=env.action_space,
        n_envs=1,
        config=swm.PlanConfig(**cfg.plan_config),
    )
    # CEMSolver.configure assumes action_space.shape == (n_envs, raw_action_dim),
    # but our raw gymnasium env reports shape (raw_action_dim,). Override the
    # derived action_dim so the solver plans over the correct 8-d joint torques.
    solver._action_dim = int(np.prod(env.action_space.shape))

    # --- Pick start/goal pairs from random episodes.
    # Goal is the frame at (start + traj_steps - 1), so an episode of length
    # traj_steps is exactly long enough: frames 0..traj_steps-1 with the goal
    # at the final frame.
    rng = np.random.default_rng(cfg.eval.seed)
    ep_lengths = raw_dataset.lengths
    ep_offsets = raw_dataset.offsets
    usable_eps = np.where(ep_lengths >= cfg.eval.traj_steps)[0]
    if len(usable_eps) < cfg.eval.num_rollouts:
        raise RuntimeError(
            f"Need {cfg.eval.num_rollouts} episodes of length >= {cfg.eval.traj_steps},"
            f" only {len(usable_eps)} available."
        )
    picked = rng.choice(usable_eps, size=cfg.eval.num_rollouts, replace=False)
    print(f"Rollouts on episodes: {picked.tolist()}")

    # --- wandb
    wandb_run = None
    if cfg.wandb.enabled:
        import wandb

        wandb_run = wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # --- Run rollouts.
    rollouts = []
    for i, ep_idx in enumerate(picked):
        print(f"\n--- rollout {i}/{cfg.eval.num_rollouts} (episode {ep_idx}) ---")
        global_start = int(ep_offsets[ep_idx])
        global_goal = global_start + cfg.eval.traj_steps - 1
        start_row = raw_dataset.get_row_data([global_start])
        goal_row = raw_dataset.get_row_data([global_goal])
        start_state = start_row["observation"][0]
        goal_state = goal_row["observation"][0]
        # pixels from get_row_data are (1, H, W, 3) uint8
        goal_frame_raw = goal_row["pixels"][0]

        result = run_rollout(
            cost_model=cost_model,
            env=env,
            solver=solver,
            start_state=start_state,
            goal_frame_raw=goal_frame_raw,
            goal_state=goal_state,
            img_transform=img_transform,
            action_scaler=action_scaler,
            action_block=cfg.plan_config.action_block,
            receding_horizon=cfg.plan_config.receding_horizon,
            traj_steps=cfg.eval.traj_steps,
            device=device,
            history_size=cfg.history_size,
        )
        print(
            f"  steps taken: {result['frames'].shape[0]} | "
            f"success_step: {result['success_step']} | "
            f"final dist to goal: {np.linalg.norm(result['xys'][-1] - result['goal_xy']):.3f}"
        )
        rollouts.append(result)

    # --- Collect val latents for the t-SNE/PCA overlay.
    print("\nCollecting val latents for t-SNE/PCA ...")
    val_latents, val_states = collect_val_latents(
        cost_model, raw_dataset, img_transform, cfg.latent_viz.num_points, device
    )

    # --- Log everything to wandb.
    if wandb_run is not None:
        log_rollout_artifacts(rollouts, val_latents, val_states, cache_dir, wandb_run)

    # --- Summary to results.txt next to the checkpoint.
    successes = [r["success_step"] != -1 for r in rollouts]
    mean_mse = float(
        np.nanmean([
            ((r["predicted_emb"] - r["actual_emb"]) ** 2).mean()
            for r in rollouts
            if r["predicted_emb"].shape[0] > 0
        ])
    )
    success_rate = float(np.mean(successes)) if successes else 0.0
    results_path = Path(cache_dir, cfg.policy).parent / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("a") as f:
        f.write("\n==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n==== RESULTS ====\n")
        f.write(f"num_rollouts: {len(rollouts)}\n")
        f.write(f"success_rate: {success_rate:.3f}\n")
        f.write(f"mean_prediction_mse: {mean_mse:.6f}\n")

    print(f"\nsuccess_rate={success_rate:.3f} mean_pred_mse={mean_mse:.6f}")
    print(f"Wrote summary to {results_path}")

    if wandb_run is not None:
        wandb_run.log({
            "eval/success_rate": success_rate,
            "eval/mean_prediction_mse": mean_mse,
        })
        wandb_run.finish()


if __name__ == "__main__":
    run()
