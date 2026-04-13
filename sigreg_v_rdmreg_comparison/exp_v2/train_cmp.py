"""Unified training script for RDMReg vs SIGReg comparison on pusht.

Picks the regularizer based on cfg.method ∈ {"rdmreg", "sigreg"} and writes
per-epoch timing + sparsity statistics to a JSON log in the run dir.
"""

import json
import os
import sys
import time
from functools import partial
from pathlib import Path

# Make the le-wm root importable so we can reuse its jepa/module/utils/rdmreg.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "sigreg_v_rdmreg_comparison"))

import hdf5plugin  # noqa: F401  — register HDF5 compression filters
import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg
from rdmreg import RDMReg
from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack


def _forward(self, batch, stage, cfg):
    """Compute next-embedding MSE + regularizer loss."""
    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    method = cfg.method

    if batch["pixels"].dtype == torch.float16:
        batch["pixels"] = batch["pixels"].to(torch.bfloat16)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)
    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]
    tgt_emb = emb[:, n_preds:]
    pred_emb = self.model.predict(ctx_emb, ctx_act)

    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()

    if method == "rdmreg":
        lambd = cfg.loss.rdmreg.weight
        output["rdmreg_loss"] = self.rdmreg(emb)
        reg_loss = output["rdmreg_loss"]
    elif method == "sigreg":
        lambd = cfg.loss.sigreg.weight
        # SIGReg expects (T, B, D)
        output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
        reg_loss = output["sigreg_loss"]
    else:
        raise ValueError(f"Unknown method {method!r}")

    output["loss"] = output["pred_loss"] + lambd * reg_loss

    # Track sparsity of the latent embeddings (fraction of near-zero entries).
    with torch.no_grad():
        z = emb.detach().float()
        output["sparsity"] = (z.abs() < 1e-5).float().mean()
        output["emb_std"] = z.std()

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k or k in ("sparsity", "emb_std")}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


class EpochTimerCallback(Callback):
    """Logs wall-clock seconds per training epoch + rolling sparsity/losses to JSON."""

    def __init__(self, json_path: Path):
        super().__init__()
        self.json_path = Path(json_path)
        self.records = []
        self._epoch_start = None

    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_start = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        elapsed = time.time() - self._epoch_start
        logged = trainer.callback_metrics
        rec = {
            "epoch": int(trainer.current_epoch + 1),
            "seconds": float(elapsed),
            "global_step": int(trainer.global_step),
        }
        for k, v in logged.items():
            if torch.is_tensor(v):
                try:
                    rec[str(k)] = float(v.item())
                except Exception:
                    continue
        self.records.append(rec)
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        with self.json_path.open("w") as f:
            json.dump(self.records, f, indent=2)


@hydra.main(version_base=None, config_path="../../config/train", config_name="lewm_cmp")
def run(cfg):
    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
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
    val = torch.utils.data.DataLoader(
        val_set, **cfg.loader, shuffle=False, drop_last=False
    )

    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )
    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    predictor = ARPredictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **cfg.predictor,
    )
    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
    projector = MLP(
        input_dim=hidden_dim, output_dim=embed_dim, hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d
    )
    predictor_proj = MLP(
        input_dim=hidden_dim, output_dim=embed_dim, hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d
    )

    world_model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
    )
    world_model = torch.compile(world_model)

    optimizers = {
        "model_opt": {
            "modules": "model",
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    module_kwargs = dict(
        model=world_model,
        forward=partial(_forward, cfg=cfg),
        optim=optimizers,
    )
    if cfg.method == "rdmreg":
        module_kwargs["rdmreg"] = RDMReg(**cfg.loss.rdmreg.kwargs)
    elif cfg.method == "sigreg":
        module_kwargs["sigreg"] = SIGReg(**cfg.loss.sigreg.kwargs)
    else:
        raise ValueError(f"Unknown method {cfg.method}")

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(**module_kwargs)

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    timer_json = run_dir / "epoch_timings.json"
    timer_cb = EpochTimerCallback(timer_json)
    object_cb = ModelObjectCallBack(dirpath=run_dir, filename=cfg.output_model_name, epoch_interval=1)

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_cb, timer_cb],
        num_sanity_val_steps=0,
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
