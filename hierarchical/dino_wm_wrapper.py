"""Wrapper around DINO-WM model loaded from official checkpoint.

Loads a DINO-WM .pth checkpoint and exposes a JEPA-like interface
(encode / predict / rollout / get_cost) compatible with our CEM solver
and the ToMe-accelerated tome_predict / tome_rollout helpers.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torchvision import transforms

# Make the dino_wm stub modules importable so the pickled checkpoint loads.
_STUB_DIR = str(Path(__file__).parent / "dino_wm_src")
if _STUB_DIR not in sys.path:
    sys.path.insert(0, _STUB_DIR)


def load_dino_wm(ckpt_path: str, device: str = "cuda"):
    """Load components from a DINO-WM checkpoint.

    Returns a dict with: encoder, predictor, action_encoder, proprio_encoder,
    decoder, encoder_image_size, image_size, action_dim, proprio_dim, num_hist,
    num_action_repeat, num_proprio_repeat, concat_dim.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    encoder = None
    if "encoder" in ckpt:
        encoder = ckpt["encoder"]
    else:
        # Encoder weights aren't always saved (DINOv2 is loaded from torch.hub).
        from models.dino import DinoV2Encoder
        encoder = DinoV2Encoder(name="dinov2_vits14", feature_key="x_norm_patchtokens")

    predictor = ckpt["predictor"]
    action_encoder = ckpt["action_encoder"]
    proprio_encoder = ckpt["proprio_encoder"]
    decoder = ckpt.get("decoder", None)

    # Hardcoded from PushT config (matches the checkpoint we just inspected).
    image_size = 224
    num_hist = 3
    num_action_repeat = 1
    num_proprio_repeat = 1
    concat_dim = 1
    action_dim = action_encoder.patch_embed.weight.shape[1]   # in_chans of Conv1d
    proprio_dim = proprio_encoder.patch_embed.weight.shape[1]
    action_emb_dim = action_encoder.patch_embed.weight.shape[0]  # out_channels
    proprio_emb_dim = proprio_encoder.patch_embed.weight.shape[0]

    # encoder_transform: resize from image_size (224) to encoder_image_size (196).
    decoder_scale = 16
    num_side_patches = image_size // decoder_scale  # 14
    encoder_image_size = num_side_patches * encoder.patch_size  # 14 * 14 = 196
    encoder_transform = transforms.Compose([transforms.Resize(encoder_image_size)])

    return DinoWMWrapper(
        encoder=encoder.to(device).eval(),
        predictor=predictor.to(device).eval(),
        action_encoder=action_encoder.to(device).eval(),
        proprio_encoder=proprio_encoder.to(device).eval(),
        decoder=decoder.to(device).eval() if decoder is not None else None,
        encoder_transform=encoder_transform,
        image_size=image_size,
        encoder_image_size=encoder_image_size,
        num_hist=num_hist,
        num_action_repeat=num_action_repeat,
        num_proprio_repeat=num_proprio_repeat,
        concat_dim=concat_dim,
        action_dim=action_emb_dim * num_action_repeat,
        proprio_dim=proprio_emb_dim * num_proprio_repeat,
        raw_action_dim=action_dim,
        raw_proprio_dim=proprio_dim,
    )


class DinoWMWrapper(nn.Module):
    """JEPA-like interface around a DINO-WM checkpoint.

    The forward path matches VWorldModel.encode / predict / rollout, but
    we expose:
        - encode(info)               : info dict in/out (LeWM-style)
        - predict(z)                 : (B, T, P, D) -> (B, T, P, D)
        - rollout(info, actions)     : autoregressive rollout for the CEM solver
        - get_cost(info, candidates) : MSE-to-goal cost for CEM
    """

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        proprio_encoder,
        decoder,
        encoder_transform,
        image_size,
        encoder_image_size,
        num_hist,
        num_action_repeat,
        num_proprio_repeat,
        concat_dim,
        action_dim,        # action_emb_dim * num_action_repeat (channels appended to z)
        proprio_dim,       # proprio_emb_dim * num_proprio_repeat
        raw_action_dim,    # input dim of action_encoder (2*frameskip for PushT)
        raw_proprio_dim,   # input dim of proprio_encoder (4 for PushT with velocity)
    ):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.proprio_encoder = proprio_encoder
        self.decoder = decoder
        self.encoder_transform = encoder_transform
        self.image_size = image_size
        self.encoder_image_size = encoder_image_size
        self.num_hist = num_hist
        self.num_action_repeat = num_action_repeat
        self.num_proprio_repeat = num_proprio_repeat
        self.concat_dim = concat_dim
        self.action_dim = action_dim
        self.proprio_dim = proprio_dim
        self.raw_action_dim = raw_action_dim
        self.raw_proprio_dim = raw_proprio_dim

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    def encode_visual(self, pixels):
        """pixels: (B, T, 3, H, W) -> (B, T, P, D_visual)."""
        B = pixels.shape[0]
        x = rearrange(pixels, "b t c h w -> (b t) c h w").float()
        x = self.encoder_transform(x)
        emb = self.encoder.forward(x)              # (B*T, P, D)
        emb = rearrange(emb, "(b t) p d -> b t p d", b=B)
        return emb

    def encode_proprio(self, proprio):
        """proprio: (B, T, raw_proprio_dim) -> (B, T, proprio_emb_dim)."""
        return self.proprio_encoder(proprio.float())

    def encode_act(self, act):
        """act: (B, T, raw_action_dim) -> (B, T, action_emb_dim)."""
        return self.action_encoder(act.float())

    def _concat(self, visual, proprio_emb, act_emb):
        """Concatenate along feature dim (concat_dim=1) — DINO-WM style.

        visual:      (B, T, P, D_visual)
        proprio_emb: (B, T, D_proprio)
        act_emb:     (B, T, D_action)
        returns:     (B, T, P, D_visual + D_proprio*repeat + D_action*repeat)
        """
        proprio_tiled = repeat(
            proprio_emb.unsqueeze(2), "b t 1 a -> b t f a", f=visual.shape[2]
        ).repeat(1, 1, 1, self.num_proprio_repeat)
        act_tiled = repeat(
            act_emb.unsqueeze(2), "b t 1 a -> b t f a", f=visual.shape[2]
        ).repeat(1, 1, 1, self.num_action_repeat)
        return torch.cat([visual, proprio_tiled, act_tiled], dim=3)

    def encode(self, info):
        """Encode an info dict into z = (B, T, P, D_full).

        info needs: 'pixels' (B, T, 3, H, W), 'proprio' (B, T, raw_proprio_dim),
        'action' (B, T, raw_action_dim).
        """
        visual = self.encode_visual(info["pixels"])
        proprio_emb = self.encode_proprio(info["proprio"])
        act_emb = self.encode_act(info["action"])
        z = self._concat(visual, proprio_emb, act_emb)
        info["z"] = z
        info["visual_emb"] = visual
        return info

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, z):
        """z: (B, T, P, D) -> (B, T, P, D)."""
        T = z.shape[1]
        z_flat = rearrange(z, "b t p d -> b (t p) d")
        z_flat = self.predictor(z_flat)
        return rearrange(z_flat, "b (t p) d -> b t p d", t=T)

    def replace_actions_from_z(self, z, act):
        """Overwrite the action slice of z with newly-encoded actions.

        z:   (B, T, P, D)
        act: (B, T, raw_action_dim)
        """
        act_emb = self.encode_act(act)
        act_tiled = repeat(
            act_emb.unsqueeze(2), "b t 1 a -> b t f a", f=z.shape[2]
        ).repeat(1, 1, 1, self.num_action_repeat)
        z = z.clone()
        z[..., -self.action_dim :] = act_tiled
        return z

    def separate_visual(self, z):
        """Strip the action+proprio slices, return only the visual feature dim."""
        return z[..., : -(self.proprio_dim + self.action_dim)]

    # ------------------------------------------------------------------
    # Rollout (LeWM-style — flattens (B, S) into batch)
    # ------------------------------------------------------------------

    def rollout(self, info, action_sequence):
        """Autoregressive rollout for the CEM solver.

        info: dict with 'pixels' (B, S, T_hist, 3, H, W),
                       'proprio' (B, S, T_hist, raw_proprio_dim),
                       'action' (B, S, T_hist, raw_action_dim) — for the
                                first num_hist steps; predicted ones are
                                generated below.
        action_sequence: (B, S, T_total, raw_action_dim). The first num_hist
                         actions are the history actions; the rest are the
                         candidate actions to roll out.
        Returns info with 'predicted_emb' = visual-only embeddings of shape
        (B, S, T_total, P, D_visual).
        """
        assert "pixels" in info, "rollout: info dict must contain 'pixels'"
        B, S = action_sequence.shape[:2]
        T_total = action_sequence.shape[2]
        T_hist = self.num_hist
        n_steps = T_total - T_hist

        # Flatten (B, S) -> (BS, ...)
        def flat(t):
            return rearrange(t, "b s ... -> (b s) ...")

        pixels = flat(info["pixels"])         # (BS, T_hist, 3, H, W)
        proprio = flat(info["proprio"])       # (BS, T_hist, raw_proprio_dim)
        actions = flat(action_sequence)       # (BS, T_total, raw_action_dim)
        act_hist = actions[:, :T_hist]
        act_future = actions[:, T_hist:]

        z = self.encode({"pixels": pixels, "proprio": proprio, "action": act_hist})["z"]
        # z is (BS, T_hist, P, D)

        for t in range(n_steps):
            z_pred = self.predict(z[:, -T_hist:])           # (BS, T_hist, P, D)
            z_new = z_pred[:, -1:]                          # (BS, 1, P, D)
            # Replace the action slice with the next planned action.
            next_act = act_future[:, t : t + 1]             # (BS, 1, raw_action_dim)
            z_new = self.replace_actions_from_z(z_new, next_act)
            z = torch.cat([z, z_new], dim=1)

        # One more predict to get the final state's prediction (matches DINO-WM rollout)
        z_pred = self.predict(z[:, -T_hist:])
        z_new = z_pred[:, -1:]
        z = torch.cat([z, z_new], dim=1)

        visual_z = self.separate_visual(z)                   # (BS, T_total+1, P, D_visual)
        info["predicted_emb"] = rearrange(visual_z, "(b s) ... -> b s ...", b=B, s=S)
        return info

    # ------------------------------------------------------------------
    # Cost (CEM solver entry point)
    # ------------------------------------------------------------------

    def set_goal(self, goal_pixels):
        """Pre-compute and cache the goal visual embedding.

        goal_pixels: (B, 3, H, W) or (B, 1, 3, H, W). Stored in
        self._goal_visual as (B, 1, P, D_visual).
        """
        if goal_pixels.ndim == 4:
            goal_pixels = goal_pixels.unsqueeze(1)
        device = next(self.parameters()).device
        with torch.no_grad():
            self._goal_visual = self.encode_visual(goal_pixels.to(device))

    def get_cost(self, info, action_candidates):
        """Rollout candidates, return MSE-to-stored-goal cost.

        info needs: 'pixels', 'proprio', 'action' — each with leading dims
        (B, S, ...) after CEM solver expansion.
        Call set_goal(...) once before invoking solver.solve().
        """
        assert hasattr(self, "_goal_visual"), "Call model.set_goal(...) first"
        device = next(self.parameters()).device
        for k in list(info.keys()):
            if torch.is_tensor(info[k]):
                info[k] = info[k].to(device)

        info = self.rollout(info, action_candidates)
        pred_visual = info["predicted_emb"]                   # (B, S, T_total+1, P, D)
        pred_last = pred_visual[:, :, -1, :, :]               # (B, S, P, D)

        # Broadcast goal across samples: (B, 1, P, D) -> (B, S, P, D)
        goal = self._goal_visual[:, -1]                       # (B, P, D)
        goal = goal.unsqueeze(1).expand_as(pred_last)
        cost = F.mse_loss(pred_last, goal, reduction="none").mean(dim=(2, 3))
        return cost  # (B, S)
