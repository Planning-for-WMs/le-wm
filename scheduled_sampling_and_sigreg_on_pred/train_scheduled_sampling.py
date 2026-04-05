import math
import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg
from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack


def inverse_sigmoid_schedule(step, k):
    """Probability of teacher forcing. Starts near 1, decays toward 0.

    p = k / (k + exp(step / k))
    """
    return k / (k + math.exp(step / k))


def lejepa_forward(self, batch, stage, cfg):
    """Encode observations, predict next states with scheduled sampling.

    Early in training the predictor receives mostly encoder outputs (teacher
    forcing).  As training progresses the predictor increasingly receives its
    own previous predictions (with stop-grad), following an inverse-sigmoid
    decay schedule.
    """

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight
    k = cfg.loss.scheduled_sampling.k

    # Upcast float16 pixels from DataLoader to bfloat16 for model
    if batch["pixels"].dtype == torch.float16:
        batch["pixels"] = batch["pixels"].to(torch.bfloat16)

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]      # (B, T, D)  T = ctx_len + n_preds
    act_emb = output["act_emb"]

    # Teacher forcing probability
    p_teacher = inverse_sigmoid_schedule(self.global_step, k)

    # Build input sequence step-by-step with scheduled sampling
    input_emb = emb[:, :ctx_len].clone()   # (B, ctx_len, D)
    input_act = act_emb[:, :ctx_len].clone()

    predictions = []
    for t in range(n_preds):
        # Predict from current input sequence; take the last output
        pred_t = self.model.predict(input_emb, input_act)[:, -1:]  # (B, 1, D)
        predictions.append(pred_t)

        if t < n_preds - 1:
            # Choose next input: encoder embedding (teacher) or own prediction (student)
            teacher_emb = emb[:, ctx_len + t : ctx_len + t + 1]
            student_emb = pred_t.detach()  # stop gradient on own predictions

            use_teacher = (
                torch.rand(emb.size(0), 1, 1, device=emb.device) < p_teacher
            )
            next_emb = torch.where(use_teacher, teacher_emb, student_emb)

            next_act = act_emb[:, ctx_len + t : ctx_len + t + 1]
            input_emb = torch.cat([input_emb, next_emb], dim=1)
            input_act = torch.cat([input_act, next_act], dim=1)

    pred_emb = torch.cat(predictions, dim=1)  # (B, n_preds, D)
    tgt_emb = emb[:, ctx_len:]                # (B, n_preds, D)

    # LeWM loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]
    output["p_teacher"] = p_teacher

    losses_dict = {
        f"{stage}/{k_}": v.detach() if torch.is_tensor(v) else v
        for k_, v in output.items()
        if "loss" in k_ or k_ == "p_teacher"
    }
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm_scheduled_sampling")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset = swm.data.LMDBDataset(**cfg.data.dataset, transform=None)
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue

            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

    # Convert preprocessed pixels to float16 to reduce DataLoader IPC overhead
    class PixelsToHalf:
        def __call__(self, x):
            x['pixels'] = x['pixels'].half()
            return x

    transforms.append(PixelsToHalf())
    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)

    ##############################
    ##       model / optim      ##
    ##############################

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
        num_frames=cfg.wm.history_size + cfg.wm.num_preds - 1,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **cfg.predictor,
    )

    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)

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

    world_model = torch.compile(world_model)

    optimizers = {
        'model_opt': {
            "modules": 'model',
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
        dirpath=run_dir, filename=cfg.output_model_name, epoch_interval=1,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
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
    return


if __name__ == "__main__":
    run()
