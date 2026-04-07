
# LeWorldModel
### Stable End-to-End Joint-Embedding Predictive Architecture from Pixels

[Lucas Maes*](https://x.com/lucasmaes_), [Quentin Le Lidec*](https://quentinll.github.io/), [Damien Scieur](https://scholar.google.com/citations?user=hNscQzgAAAAJ&hl=fr), [Yann LeCun](https://yann.lecun.com/) and [Randall Balestriero](https://randallbalestriero.github.io/)

**Abstract:** Joint Embedding Predictive Architectures (JEPAs) offer a compelling framework for learning world models in compact latent spaces, yet existing methods remain fragile, relying on complex multi-term losses, exponential moving averages, pretrained encoders, or auxiliary supervision to avoid representation collapse. In this work, we introduce LeWorldModel (LeWM), the first JEPA that trains stably end-to-end from raw pixels using only two loss terms: a next-embedding prediction loss and a regularizer enforcing Gaussian-distributed latent embeddings. This reduces tunable loss hyperparameters from six to one compared to the only existing end-to-end alternative. With ~15M parameters trainable on a single GPU in a few hours, LeWM plans up to 48× faster than foundation-model-based world models while remaining competitive across diverse 2D and 3D control tasks. Beyond control, we show that LeWM's latent space encodes meaningful physical structure through probing of physical quantities. Surprise evaluation confirms that the model reliably detects physically implausible events.

<p align="center">
   <b>[ <a href="https://arxiv.org/pdf/2603.19312v1">Paper</a> | <a href="https://drive.google.com/drive/folders/1r31os0d4-rR0mdHc7OlY_e5nh3XT4r4e?usp=sharing">Checkpoints</a> | <a href="https://huggingface.co/collections/quentinll/lewm">Data</a> | <a href="https://le-wm.github.io/">Website</a> ]</b>
</p>

<br>

<p align="center">
  <img src="assets/lewm.gif" width="80%">
</p>

If you find this code useful, please reference it in your paper:
```
@article{maes_lelidec2026lewm,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint},
  year={2026}
}
```

## Using the code
This codebase builds on [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) for environment management, planning, and evaluation, and [stable-pretraining](https://github.com/galilai-group/stable-pretraining) for training. Together they reduce this repository to its core contribution: the model architecture and training objective.

**Installation:**
```bash
uv venv --python=3.10
source .venv/bin/activate
uv pip install stable-worldmodel[train,env]
```

## Data

Datasets use the HDF5 format for fast loading. Download the data from [HuggingFace](https://huggingface.co/collections/quentinll/lewm) and decompress with:

```bash
tar --zstd -xvf archive.tar.zst
```

Place the extracted `.h5` files under `$STABLEWM_HOME` (defaults to `~/.stable-wm/`). You can override this path:
```bash
export STABLEWM_HOME=/path/to/your/storage
```

Dataset names are specified without the `.h5` extension. For example, `config/train/data/pusht.yaml` references `pusht_expert_train`, which resolves to `$STABLEWM_HOME/pusht_expert_train.h5`.

## Antmaze from OGBench (explore split)

LeWM can be trained end-to-end from pixels on OGBench locomaze datasets. The flagship locomotion task in this repo is `antmaze-medium-explore-v0`: a 4-legged MuJoCo ant exploring a medium-size maze. The world model has to learn **both** ant body dynamics (8 joint torques → limb and body motion) **and** 2D navigation, directly from rendered frames.

OGBench ships the dataset as a single `.npz` of 29-d MuJoCo states (`qpos + qvel`) and 8-d actions — 10k episodes × 500 steps (≈5M transitions). Because LeWM trains on pixels, we **render 112×112 frames offline** from the saved states via MuJoCo and store them in an LMDB that the existing `LMDBDataset` loader can consume unchanged. The render pass is a one-off cost; after that the LMDB streams frames at the same speed as the other tasks.

### 1. Data preparation

```bash
source le-wm/bin/activate
MUJOCO_GL=egl python convert_ogbench_antmaze.py --max_episodes 200
```

Flags:
- `--dataset_name antmaze-medium-explore-v0` — any OGBench locomaze id with state observations.
- `--max_episodes 200` — subset size for fast iteration (≈100k frames, ~0.5 GB LMDB). Pass `0` to convert all 10k episodes.
- `--out_path dataset/antmaze_medium_explore.lmdb` — output directory.
- `--img_size 112 --jpeg_quality 95 --encoding jpeg` — matches the rest of the repo.

The script (a) downloads the `.npz` on first run to `~/.ogbench/data/`, (b) groups valid rows into episodes using the OGBench compact-dataset `valids` field (dropping trajectory-boundary frames), (c) calls `env.set_state(qpos, qvel); env.render(...)` per step, and (d) writes JPEG-encoded frames plus per-frame `action`/`observation` metadata in the existing le-wm LMDB schema.

### 2. Training

```bash
python train_antmaze_explore.py
```

Config: [`config/train/lewm_antmaze.yaml`](config/train/lewm_antmaze.yaml). Key choices:

| | |
|---|---|
| Encoder | ViT-tiny (hidden=192, patch=14) at 112×112 — ~5M params |
| Predictor | `depth=3, heads=4, mlp_dim=512, dim_head=32` — ~1.5M params (tiny) |
| History | `history_size=1` (single frame + 5-step action block) |
| Frameskip | 5 (matches pusht/cube lewm configs) |
| Loss | vanilla LeWM: MSE next-embedding + SIGReg (one λ) |
| Epochs | 50 |
| Batch | 128 |

Total world model ≈ 9M params. The script logs per-module parameter counts at startup. If the MSE-vs-rollout-step curve in eval shows the tiny predictor underfitting long horizons, bump to medium:

```bash
python train_antmaze_explore.py predictor.depth=6 predictor.mlp_dim=2048 predictor.dim_head=64
```

A **`LatentVizCallback`** runs every 5 epochs, encoding a batch of val-set frames and logging PCA + t-SNE scatter plots of the latent space (coloured by agent x-position) to wandb. This lets you see the latent geometry organising itself without having to run eval.

Checkpoints are written to `$STABLEWM_HOME/<run_id>/lewm_antmaze_epoch_<N>_object.ckpt`.

### 3. Closed-loop MPC evaluation with visualizations

```bash
MUJOCO_GL=egl python eval_antmaze_explore.py policy=<run_id>/lewm_antmaze_epoch_50
```

Config: [`config/eval/antmaze.yaml`](config/eval/antmaze.yaml). What it does:

- Samples **3 rollouts** (configurable via `eval.num_rollouts`) from episodes in the LMDB, using each episode's first frame as the start state and frame-500 as the goal image.
- Drives the raw OGBench `antmaze-medium-v0` env in Python (no `swm.World` wrapper, to keep the logging loop transparent).
- Runs closed-loop CEM MPC: plan horizon = 10 predictor steps (= 50 env steps of lookahead, at frameskip 5), execute the first 2 plan steps (= **10 env steps**) before replanning. Total 500 env steps per rollout.
- Logs to wandb:
  - `rollout/video_{0,1,2}` — RGB videos of the full 500-step rollout with the goal frame overlaid in the top-right corner.
  - `rollout/mse_vs_step` — per-step MSE between the predictor's imagined latent and the encoder's latent for the actually-observed frame, averaged across the 3 rollouts. This is the most direct read on predictor capacity.
  - `latent/pca_with_rollouts` — PCA(2) of ~2000 val-set latents (background, coloured by x-position) with the rollout latents overlaid as line+scatter trajectories.
  - `latent/tsne_with_rollouts` — t-SNE of the combined (val + rollout) latent set with rollout trajectories highlighted.
  - `eval/success_rate`, `eval/mean_prediction_mse` — scalar summaries.
- Also writes `$STABLEWM_HOME/<run_id>/antmaze_results.txt` appending per-run config + metrics for offline comparison.

Replan frequency, horizon, and number of CEM samples are all in `config/eval/antmaze.yaml`; see [`config/eval/solver/cem.yaml`](config/eval/solver/cem.yaml) for solver hyperparameters.

## Training

`jepa.py` contains the PyTorch implementation of LeWM. Training is configured via [Hydra](https://hydra.cc/) config files under `config/train/`.

Before training, set your WandB `entity` and `project` in `config/train/lewm.yaml`:
```yaml
wandb:
  config:
    entity: your_entity
    project: your_project
```

To launch training:
```bash
python train.py data=pusht
```

Checkpoints are saved to `$STABLEWM_HOME` upon completion.

For baseline scripts, see the stable-worldmodel [scripts](https://github.com/galilai-group/stable-worldmodel/tree/main/scripts/train) folder.

## Planning

Evaluation configs live under `config/eval/`. Set the `policy` field to the checkpoint path **relative to `$STABLEWM_HOME`**, without the `_object.ckpt` suffix:

```bash
# ✓ correct
python eval.py --config-name=pusht.yaml policy=pusht/lewm

# ✗ incorrect
python eval.py --config-name=pusht.yaml policy=pusht/lewm_object.ckpt
```

## Pretrained Checkpoints

Pre-trained checkpoints are available on [Google Drive](https://drive.google.com/drive/folders/1r31os0d4-rR0mdHc7OlY_e5nh3XT4r4e). Download the checkpoint archive and place the extracted files under `$STABLEWM_HOME/`.

<div align="center">

| Method | two-room | pusht | cube | reacher |
|:---:|:---:|:---:|:---:|:---:|
| pldm | ✓ | ✓ | ✓ | ✓ |
| lejepa | ✓ | ✓ | ✓ | ✓ |
| ivl | ✓ | ✓ | ✓ | — |
| iql | ✓ | ✓ | ✓ | — |
| gcbc | ✓ | ✓ | ✓ | — |
| dinowm | ✓ | ✓ | — | — |
| dinowm_noprop | ✓ | ✓ | ✓ | ✓ |

</div>

## Loading a checkpoint

Each tar archive contains two files per checkpoint:
- `<name>_object.ckpt` — a serialized Python object for convenient loading; this is what `eval.py` and the `stable_worldmodel` API use
- `<name>_weight.ckpt` — a weights-only checkpoint (`state_dict`) for cases where you want to load weights into your own model instance

To load the object checkpoint via the `stable_worldmodel` API:

```python
import stable_worldmodel as swm

# Load the cost model (for MPC)
cost = swm.policy.AutoCostModel('pusht/lewm')
```

This function accepts:
- `run_name` — checkpoint path **relative to `$STABLEWM_HOME`**, without the `_object.ckpt` suffix
- `cache_dir` — optional override for the checkpoint root (defaults to `$STABLEWM_HOME`)

The returned module is in `eval` mode with its PyTorch weights accessible via `.state_dict()`.

## Contact & Contributions
Feel free to open [issues](https://github.com/lucas-maes/le-wm/issues)! For questions or collaborations, please contact `lucas.maes@mila.quebec`
