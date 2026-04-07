"""Train LeWM on og-bench antmaze-medium-explore-v0.

The model/loss wiring mirrors ``train.py`` (same JEPA, same SIGReg-regularised
prediction loss). Extras on top of the base pipeline:
    (a) ``LatentVizCallback`` — t-SNE + PCA of encoder latents, logged every
        N epochs so the latent geometry can be inspected during training,
    (b) ``spt.callbacks.RankMe`` — effective-rank diagnostic for the encoder,
        a direct collapse detector.
Configured via ``config/train/lewm_antmaze.yaml``.
"""

from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg
from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack



def lejepa_forward(self, batch, stage, cfg):
    """Encode observations, predict next states, compute losses."""
    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight

    if batch["pixels"].dtype == torch.float16:
        batch["pixels"] = batch["pixels"].to(torch.bfloat16)

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D) — mean-pooled over patches
    emb_patches = output["emb_patches"]  # (B, T, N, D) — per-patch
    act_emb = output["act_emb"]

    # Flattened (B, D) view of the first-context embedding, consumed by the
    # RankMe callback (which expects a 2-D [N, D] target to SVD on).
    output["emb_flat"] = emb[:, 0].detach().float()

    # Predictor operates on per-patch embeddings: predict each patch of the
    # next frame(s) from each patch of the context frame(s) + action.
    ctx_patches = emb_patches[:, :ctx_len]    # (B, ctx_len, N, D)
    ctx_act = act_emb[:, :ctx_len]            # (B, ctx_len, D_act)

    tgt_patches = emb_patches[:, n_preds:].detach()  # (B, T', N, D)
    pred_patches = self.model.predict(ctx_patches, ctx_act)  # (B, T', N, D)

    output["pred_loss"] = (pred_patches - tgt_patches).pow(2).mean()

    # SIGReg on all patch tokens (not pooled). Reshape (B, T, N, D) to
    # (T*N, B, D) so SIGReg checks Gaussianity across the batch for each
    # (time-step, patch-position) independently, then averages.
    B, T, N, D = emb_patches.shape
    sigreg_input = emb_patches.permute(1, 2, 0, 3).reshape(T * N, B, D)
    output["sigreg_loss"] = self.sigreg(sigreg_input)
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


class LatentVizCallback(Callback):
    """Every N epochs, project encoder latents to 2D (t-SNE + PCA) and log to wandb.

    Colours t-SNE by agent xy-position taken from the ``observation`` column
    (qpos[0:2]). PCA is used as a cheap sanity-check projection that is stable
    across epochs.
    """

    def __init__(self, every_n_epochs: int = 5, num_points: int = 2000,
                 tsne_perplexity: int = 30):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.num_points = num_points
        self.tsne_perplexity = tsne_perplexity

    @torch.no_grad()
    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return
        if not trainer.is_global_zero:
            return
        logger = trainer.logger
        if logger is None or not isinstance(logger, WandbLogger):
            return

        val_loader = trainer.datamodule.val_dataloader() if hasattr(trainer, "datamodule") else None
        if val_loader is None:
            return

        device = pl_module.device
        model = pl_module.model
        was_training = model.training
        model.eval()

        embs, xys = [], []
        collected = 0
        for batch in val_loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            batch["action"] = torch.nan_to_num(batch.get("action", torch.zeros(1, device=device)), 0.0)
            if batch["pixels"].dtype in (torch.float16,):
                batch["pixels"] = batch["pixels"].to(torch.bfloat16)
            out = model.encode({k: v for k, v in batch.items()})
            emb = out["emb"][:, 0].float().cpu().numpy()  # (B, D)
            embs.append(emb)
            if "observation" in batch:
                # observation is already standard-scaled, so we use it only as
                # a stable colour, not for interpretation.
                xy = batch["observation"][:, 0, :2].float().cpu().numpy()
                xys.append(xy)
            collected += emb.shape[0]
            if collected >= self.num_points:
                break

        if was_training:
            model.train()

        if not embs:
            return

        z = np.concatenate(embs, axis=0)[: self.num_points]
        colour = (
            np.concatenate(xys, axis=0)[: self.num_points]
            if xys else np.zeros((z.shape[0], 2))
        )

        # --- PCA (stable, cheap)
        pca = PCA(n_components=2)
        z_pca = pca.fit_transform(z)
        fig_pca, ax = plt.subplots(figsize=(5, 5))
        sc = ax.scatter(z_pca[:, 0], z_pca[:, 1], c=colour[:, 0], cmap="viridis", s=6)
        plt.colorbar(sc, ax=ax, label="standardised x")
        ax.set_title(f"PCA of encoder latents (epoch {trainer.current_epoch + 1})")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        fig_pca.tight_layout()

        # --- t-SNE (slower but reveals nonlinear structure)
        perplexity = min(self.tsne_perplexity, max(5, z.shape[0] // 4))
        tsne = TSNE(n_components=2, perplexity=perplexity, init="pca",
                    random_state=0, learning_rate="auto")
        z_tsne = tsne.fit_transform(z)
        fig_tsne, ax = plt.subplots(figsize=(5, 5))
        sc = ax.scatter(z_tsne[:, 0], z_tsne[:, 1], c=colour[:, 0], cmap="viridis", s=6)
        plt.colorbar(sc, ax=ax, label="standardised x")
        ax.set_title(f"t-SNE of encoder latents (epoch {trainer.current_epoch + 1})")
        fig_tsne.tight_layout()

        # Log at trainer.global_step so wandb places these images on the same
        # step timeline as training metrics — one panel per key, slider by step.
        import wandb
        logger.experiment.log(
            {
                "latent/pca": wandb.Image(fig_pca),
                "latent/tsne": wandb.Image(fig_tsne),
            },
            step=trainer.global_step,
        )
        plt.close(fig_pca)
        plt.close(fig_tsne)


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm_antmaze")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset = swm.data.LMDBDataset(**cfg.data.dataset, transform=None)
    transforms = [get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)
            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

    class PixelsToHalf:
        def __call__(self, x):
            x["pixels"] = x["pixels"].half()
            return x

    transforms.append(PixelsToHalf())
    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(
        train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen
    )
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)

    ##############################
    ##       model / optim      ##
    ##############################

    # Any ViTConfig field set under `encoder_kwargs:` in the yaml is forwarded
    # here and overrides the preset baked into `encoder_scale`.
    encoder_extra = cfg.get("encoder_kwargs") or {}
    if not isinstance(encoder_extra, dict):
        encoder_extra = OmegaConf.to_container(encoder_extra, resolve=True) or {}
    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
        **encoder_extra,
    )

    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim
    num_patches = (cfg.img_size // cfg.patch_size) ** 2  # e.g. (224/14)^2 = 256

    predictor = ARPredictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        num_patches=num_patches,
        **cfg.predictor,
    )

    act_enc_cfg = cfg.get("action_encoder", {})
    action_encoder = Embedder(
        input_dim=effective_act_dim,
        smoothed_dim=act_enc_cfg.get("smoothed_dim", 64),
        emb_dim=embed_dim,
        mlp_scale=act_enc_cfg.get("mlp_scale", 4),
        mlp_depth=act_enc_cfg.get("mlp_depth", 3),
    )

    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    predictor_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    world_model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
    )

    # Report parameter counts so the 'small predictor' choice is visible.
    def _count(m):
        return sum(p.numel() for p in m.parameters())
    print(f"[param count] encoder={_count(encoder)/1e6:.2f}M "
          f"predictor={_count(predictor)/1e6:.2f}M "
          f"action_encoder={_count(action_encoder)/1e6:.2f}M "
          f"projector={_count(projector)/1e6:.2f}M "
          f"pred_proj={_count(predictor_proj)/1e6:.2f}M "
          f"total={_count(world_model)/1e6:.2f}M")

    # torch.compile wraps the JEPA in an OptimizedModule whose attribute
    # forwarding is fragile for the in-process eval callback (.encode,
    # .rollout, .get_cost must work on the *live* model mid-training).
    # Default off here; flip ``compile_model: true`` in the yaml if you do
    # NOT use the periodic eval callback and want the speedup.
    if cfg.get("compile_model", False):
        world_model = torch.compile(world_model)

    optimizers = {
        "model_opt": {
            "modules": "model",
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir,
        filename=cfg.output_model_name,
        epoch_interval=cfg.get("ckpt_every_n_epochs", 5),
    )

    callbacks = [object_dump_callback]

    if cfg.get("latent_viz", {}).get("enabled", False):
        callbacks.append(
            LatentVizCallback(
                every_n_epochs=cfg.latent_viz.every_n_epochs,
                num_points=cfg.latent_viz.num_points,
                tsne_perplexity=cfg.latent_viz.tsne_perplexity,
            )
        )

    # RankMe: effective-rank diagnostic on encoder latents. Low rankme values
    # (close to 1) during training are a direct collapse signal.
    if cfg.get("rankme", {}).get("enabled", False):
        callbacks.append(
            spt.callbacks.RankMe(
                name="rankme/emb",
                target="emb_flat",
                queue_length=cfg.rankme.get("queue_length", 2048),
                target_shape=cfg.wm.embed_dim,
            )
        )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=run_dir / f"{cfg.output_model_name}_weights.ckpt",
    )

    manager()


if __name__ == "__main__":
    run()
