"""JEPA Implementation"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v

class JEPA(nn.Module):

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()

    def encode(self, info):
        """Encode observations and actions into embeddings.
        info: dict with pixels and action keys

        Outputs stored in ``info``:
            emb_patches : (B, T, N_patches, embed_dim) — per-patch projected
                          embeddings (used by SIGReg which checks Gaussianity
                          at the patch level, not pooled).
            emb         : (B, T, embed_dim) — mean-pooled over patches (used
                          by the predictor, rollout, and cost functions which
                          operate at the frame level).
        """

        pixels = info['pixels'].float()
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...") # flatten for encoding
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        patches = output.last_hidden_state[:, 1:]  # all patch tokens, skip CLS — (B*T, N, hidden)
        bt, n, h = patches.shape
        proj_patches = self.projector(patches.reshape(bt * n, h))  # (B*T*N, embed_dim)
        proj_patches = proj_patches.reshape(bt, n, -1)  # (B*T, N, embed_dim)
        info["emb_patches"] = rearrange(proj_patches, "(b t) n d -> b t n d", b=b)
        info["emb"] = rearrange(proj_patches.mean(dim=1), "(b t) d -> b t d", b=b)

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])

        return info

    def predict(self, emb, act_emb):
        """Predict next state embedding.

        Supports both frame-level and patch-level prediction depending on
        the predictor's ``num_patches`` setting:

        Frame-level (legacy):
            emb: (B, T, D), act_emb: (B, T, A_emb) → returns (B, T, D)

        Patch-level:
            emb: (B, T, N, D), act_emb: (B, T, A_emb) → returns (B, T, N, D)
            pred_proj is applied per-patch.
        """
        preds = self.predictor(emb, act_emb)

        if preds.ndim == 4:
            # Patch-level: (B, T, N, D_hidden)
            B, T, N, D = preds.shape
            preds = self.pred_proj(preds.reshape(B * T * N, D))
            preds = preds.reshape(B, T, N, -1)
        else:
            # Frame-level: (B, T, D_hidden)
            preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
            preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return preds

    ####################
    ## Inference only ##
    ####################

    def rollout(self, info, action_sequence, history_size: int = 3):
        """Rollout the model given an initial info dict and action sequence.
        pixels: (B, S, T, C, H, W)
        action_sequence: (B, S, T, action_dim)
         - S is the number of action plan samples
         - T is the time horizon

        Works at patch level when the predictor has ``num_patches > 0``:
        the autoregressive loop carries forward per-patch embeddings and
        pools only at the end for the cost function (``predicted_emb``).
        """

        assert "pixels" in info, "pixels not in info_dict"
        H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        n_steps = T - H

        # copy and encode initial info dict
        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = self.encode(_init)

        patch_level = self.predictor.num_patches > 0

        if patch_level:
            # Use per-patch embeddings: (B, T_hist, N, D)
            emb_patches = _init["emb_patches"].unsqueeze(1).expand(B, S, -1, -1, -1)
            emb_patches = rearrange(emb_patches, "b s ... -> (b s) ...").clone()
        # Always keep a pooled track for the cost function.
        emb = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)
        info["emb"] = emb
        _init = {k: detach_clone(v) for k, v in _init.items()}

        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        # rollout predictor autoregressively for n_steps + 1 (the last predict)
        HS = history_size
        for t in range(n_steps + 1):
            act_emb = self.action_encoder(act)

            if patch_level:
                emb_trunc = emb_patches[:, -HS:]      # (BS, HS, N, D)
                act_trunc = act_emb[:, -HS:]           # (BS, HS, A_emb)
                pred = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, N, D)
                emb_patches = torch.cat([emb_patches, pred], dim=1)
                # Pool for the frame-level track.
                pred_pooled = pred.mean(dim=2)         # (BS, 1, D)
            else:
                emb_trunc = emb[:, -HS:]               # (BS, HS, D)
                act_trunc = act_emb[:, -HS:]
                pred_pooled = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)

            emb = torch.cat([emb, pred_pooled], dim=1)

            if t < n_steps:
                next_act = act_future[:, t : t + 1, :]
                act = torch.cat([act, next_act], dim=1)

        # unflatten batch and sample dimensions
        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout

        return info

    def criterion(self, info_dict: dict):
        """Compute the cost between predicted embeddings and goal embeddings."""
        pred_emb = info_dict["predicted_emb"]  # (B,S, T-1, dim)
        goal_emb = info_dict["goal_emb"]  # (B, S, T, dim)

        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)

        # return last-step cost per action candidate
        cost = F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction="none",
        ).sum(dim=tuple(range(2, pred_emb.ndim)))  # (B, S)

        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """ Compute the cost of action candidates given an info dict with goal and initial state."""

        assert "goal" in info_dict, "goal not in info_dict"

        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]

        for k in info_dict:
            if k.startswith("goal_"):
                goal[k[len("goal_") :]] = goal.pop(k)

        goal.pop("action")
        goal = self.encode(goal)

        info_dict["goal_emb"] = goal["emb"].unsqueeze(1)  # (B, 1, T, D) — S dim added for broadcasting
        info_dict = self.rollout(info_dict, action_candidates)

        cost = self.criterion(info_dict)
        
        return cost
