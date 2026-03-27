# LeWM-Bisim Training Config Reference

Reference for all hyperparameters in `lewm_bisim.yaml`. Parameters shared with the base `lewm.yaml` are marked accordingly.

## General

| Parameter | Default | Description |
|-----------|---------|-------------|
| `output_model_name` | `lewm_bisim` | Name used for checkpoint files and WandB run name. Base LeWM uses `lewm`. |
| `subdir` | `${hydra:job.id}` | Subdirectory under the cache dir for this run's outputs. Auto-set by Hydra. |
| `num_workers` | `6` | Number of dataloader worker processes. Referenced by `loader.num_workers`. |
| `train_split` | `0.9` | Fraction of the dataset used for training; the rest is validation. |
| `seed` | `3072` | Random seed for dataset splitting and dataloader shuffling. |
| `dump_object` | `True` | Whether to save the full model object (via `torch.save`) at each epoch. |

## Data / Preprocessing

| Parameter | Default | Description |
|-----------|---------|-------------|
| `defaults.data` | `pusht` | Which dataset config to load from `config/train/data/`. Options: `pusht`, `dmc`, `ogb`, `tworoom`. Override with `data=dmc` on the command line. |
| `img_size` | `224` | Input image resolution (pixels are resized to `img_size x img_size`). |
| `patch_size` | `14` | ViT patch size. With `img_size=224`, this gives 16x16 = 256 patches. |
| `encoder_scale` | `tiny` | ViT model scale. `tiny` corresponds to ViT-Ti (~5.7M params, hidden_size=192). |

## Trainer (Lightning)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `trainer.max_epochs` | `100` | Total number of training epochs. |
| `trainer.devices` | `auto` | Number of GPUs. `auto` uses all available. |
| `trainer.accelerator` | `gpu` | Hardware accelerator type. |
| `trainer.precision` | `bf16` | Mixed precision mode. `bf16` uses bfloat16 for faster training with minimal accuracy loss. |
| `trainer.gradient_clip_val` | `1.0` | Maximum gradient norm. Gradients are clipped to this value to prevent training instability. |

## DataLoader

| Parameter | Default | Description |
|-----------|---------|-------------|
| `loader.batch_size` | `128` | Number of trajectory sequences per batch. |
| `loader.num_workers` | `${num_workers}` (6) | Dataloader workers (references top-level `num_workers`). |
| `loader.persistent_workers` | `True` | Keep worker processes alive between epochs to avoid respawn overhead. |
| `loader.prefetch_factor` | `3` | Number of batches each worker prefetches. |
| `loader.pin_memory` | `True` | Pin batch tensors in CPU memory for faster GPU transfer. |

## Optimizer

| Parameter | Default | Description |
|-----------|---------|-------------|
| `optimizer.type` | `AdamW` | Optimizer class. AdamW applies weight decay correctly (decoupled from gradient). |
| `optimizer.lr` | `5e-5` | Base learning rate. Scheduled via `LinearWarmupCosineAnnealingLR` (linear warmup then cosine decay). |
| `optimizer.weight_decay` | `1e-3` | L2 regularization coefficient applied to all parameters. |

## World Model Architecture

| Parameter | Default | Description |
|-----------|---------|-------------|
| `wm.type` | `lewm` | Model type identifier. |
| `wm.history_size` | `3` | Number of context frames the predictor conditions on. Combined with `num_preds`, determines the total sequence length: `num_steps = history_size + num_preds = 4`. |
| `wm.num_preds` | `1` | Prediction offset. The predictor forecasts `num_preds` steps ahead. Target embeddings start at index `num_preds` in the sequence. |
| `wm.embed_dim` | `192` | Latent embedding dimension. All embeddings (state, action, prediction) are projected to this size. Matches ViT-Ti hidden size. |

## Predictor (Autoregressive Transformer)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `predictor.depth` | `6` | Number of transformer blocks in the predictor. |
| `predictor.heads` | `16` | Number of attention heads per block. |
| `predictor.mlp_dim` | `2048` | Hidden dimension of the feedforward layers within each block. |
| `predictor.dim_head` | `64` | Dimension per attention head. Total attention dimension = `heads * dim_head = 1024`. |
| `predictor.dropout` | `0.1` | Dropout rate in attention and feedforward layers. |
| `predictor.emb_dropout` | `0.0` | Dropout applied to input embeddings + positional encodings before the transformer. |

## Loss

### SIGReg (Sketch Isotropic Gaussian Regularizer)

Prevents representation collapse by enforcing that latent embeddings follow an isotropic Gaussian distribution.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `loss.sigreg.weight` | `0.09` | Lambda coefficient for SIGReg loss. Total contribution: `weight * sigreg_loss`. This is the single tunable hyperparameter of base LeWM. |
| `loss.sigreg.kwargs.knots` | `17` | Number of grid points for the Epps-Pulley goodness-of-fit statistic. More knots = finer approximation of the Gaussian test. |
| `loss.sigreg.kwargs.num_proj` | `1024` | Number of random projections used to estimate the distribution. Higher values reduce variance of the estimate. |

### Bisim (Bisimulation Metric Loss) -- *new in lewm_bisim*

Enforces that embedding distances between consecutive states match the magnitude of the action taken between them: `L_bisim = (||E(x_i) - sg(E(x_{i-1}))||_2 - ||sg(a_{i-1})||_2)^2`. This produces a locally-straightened latent space where distances are meaningful in terms of control effort, simplifying planning.

- `sg(·)` denotes stop-gradient (`.detach()`). The previous state embedding and the action embedding are frozen -- gradients only flow through the current state's encoder.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `loss.bisim.weight` | `1.0` | Target lambda for bisim loss after warmup completes. Total contribution: `effective_lambda * bisim_loss`. Set to `0.0` to disable (reduces to base LeWM). |
| `loss.bisim.warmup_epochs` | `10` | Number of epochs over which `effective_lambda` linearly ramps from `0` to `weight`. This avoids destabilizing early training when the encoder and action embedder are still learning meaningful representations. Set to `0` to apply full weight from epoch 0. |

**Warmup schedule:** At epoch `e`, the effective lambda is `weight * min(e / warmup_epochs, 1.0)`.

## WandB Logging

| Parameter | Default | Description |
|-----------|---------|-------------|
| `wandb.enabled` | `True` | Enable Weights & Biases logging. |
| `wandb.config.entity` | `lewm` | WandB team/user. |
| `wandb.config.project` | `lewm` | WandB project name. |
| `wandb.config.name` | `${output_model_name}` | Run display name. |
| `wandb.config.id` | `${subdir}` | Run ID for resuming. |
| `wandb.config.resume` | `allow` | Resume behavior if a run with the same ID exists. |
| `wandb.config.log_model` | `False` | Whether to upload model artifacts to WandB. |

## Total Loss

The complete training objective is:

```
L = L_pred + λ_sigreg * L_sigreg + λ_bisim(e) * L_bisim
```

where:
- `L_pred`: MSE between predicted and target embeddings
- `L_sigreg`: SIGReg regularization (`loss.sigreg.weight = 0.09`)
- `L_bisim`: Bisimulation metric loss (`loss.bisim.weight = 1.0`, with linear warmup)
- `λ_bisim(e)`: Epoch-dependent weight with linear warmup over `loss.bisim.warmup_epochs`

## Usage Examples

```bash
# Default (PushT dataset)
python train_bisim.py

# Different dataset
python train_bisim.py data=dmc

# Tune bisim hyperparameters
python train_bisim.py loss.bisim.weight=0.5 loss.bisim.warmup_epochs=20

# Disable bisim (equivalent to base LeWM)
python train_bisim.py loss.bisim.weight=0.0

# No warmup (full bisim weight from epoch 0)
python train_bisim.py loss.bisim.warmup_epochs=0

# Override multiple settings
python train_bisim.py data=dmc trainer.max_epochs=200 optimizer.lr=1e-4 loss.bisim.weight=2.0
```
