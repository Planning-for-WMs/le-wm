"""ToMe-accelerated prediction and rollout for DINO-WM.

Uses token merging (facebook/tome) to speed up the DINO-WM ViTPredictor
during planning. All merge/unmerge logic comes from tome.merge — nothing
is rewritten.
"""

from contextlib import contextmanager

import torch
from einops import rearrange

from tome.merge import bipartite_soft_matching, merge_wavg
from tome.utils import parse_r


@contextmanager
def _disable_causal_mask(predictor):
    """Temporarily replace each Attention.bias with all-ones so the
    masked_fill becomes a no-op. ToMe shrinks the sequence between layers,
    which invalidates the precomputed block-causal mask: the simplest fix
    is to attend over the entire merged sequence.
    """
    saved = []
    for layer in predictor.transformer.layers:
        attn = layer[0]                       # (Attention, FeedForward)
        saved.append((attn, attn.bias))
        attn.bias = torch.ones_like(attn.bias)
    try:
        yield
    finally:
        for attn, b in saved:
            attn.bias = b


def tome_predict(predictor, z_flat, r):
    """Run the DINO-WM ViTPredictor with ToMe token merging between blocks.

    Args:
        predictor: ViTPredictor instance (from dino_wm_src.models.vit)
        z_flat: (B, T*P, D) flattened token sequence
        r: int or list — tokens to merge per layer (passed to parse_r)

    Returns:
        (B, T*P, D) — unmerged output at original resolution
    """
    transformer = predictor.transformer

    # 1. Pos embedding + dropout (same as ViTPredictor.forward)
    n = z_flat.shape[1]
    x = z_flat + predictor.pos_embedding[:, :n]
    x = predictor.dropout(x)

    # 2. Per-layer schedule
    r_schedule = parse_r(len(transformer.layers), r)

    unmerge_fns = []
    size = None
    with _disable_causal_mask(predictor):
        for (attn, ff), ri in zip(transformer.layers, r_schedule):
            # Standard ViTPredictor block: attn(x) + x, then ff(x) + x.
            x = attn(x) + x
            x = ff(x) + x
            if ri > 0:
                merge, unmerge = bipartite_soft_matching(x, ri)
                x, size = merge_wavg(merge, x, size)
                unmerge_fns.append(unmerge)
        x = transformer.norm(x)

    # 3. Unmerge in reverse to recover the original token count
    for unmerge in reversed(unmerge_fns):
        x = unmerge(x)

    return x


def tome_rollout(model, info, action_sequence, r):
    """Autoregressive rollout using ToMe-accelerated prediction.

    Mirrors DinoWMWrapper.rollout but calls tome_predict in place of
    model.predict.
    """
    assert "pixels" in info
    B, S = action_sequence.shape[:2]
    T_total = action_sequence.shape[2]
    T_hist = model.num_hist
    n_steps = T_total - T_hist

    def flat(t):
        return rearrange(t, "b s ... -> (b s) ...")

    pixels = flat(info["pixels"])
    proprio = flat(info["proprio"])
    actions = flat(action_sequence)
    act_hist = actions[:, :T_hist]
    act_future = actions[:, T_hist:]

    z = model.encode({"pixels": pixels, "proprio": proprio, "action": act_hist})["z"]
    P = z.shape[2]

    def _tome_predict_4d(z_in):
        # z_in: (BS, T_hist, P, D) -> flatten -> tome_predict -> unflatten
        T = z_in.shape[1]
        z_flat = rearrange(z_in, "b t p d -> b (t p) d")
        z_out = tome_predict(model.predictor, z_flat, r)
        return rearrange(z_out, "b (t p) d -> b t p d", t=T, p=P)

    for t in range(n_steps):
        z_pred = _tome_predict_4d(z[:, -T_hist:])
        z_new = z_pred[:, -1:]
        next_act = act_future[:, t : t + 1]
        z_new = model.replace_actions_from_z(z_new, next_act)
        z = torch.cat([z, z_new], dim=1)

    z_pred = _tome_predict_4d(z[:, -T_hist:])
    z_new = z_pred[:, -1:]
    z = torch.cat([z, z_new], dim=1)

    visual_z = model.separate_visual(z)
    info["predicted_emb"] = rearrange(visual_z, "(b s) ... -> b s ...", b=B, s=S)
    return info
