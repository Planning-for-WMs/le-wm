"""DINO-WM predictor: standard transformer (no AdaLN action conditioning).

Actions are added to patch embeddings before the transformer, rather than
modulating internal blocks via AdaLN as in LeWM's ARPredictor.
"""

import torch
from torch import nn
from einops import rearrange

from module import Block, Transformer


class DINOPredictor(nn.Module):
    """Patch-level predictor for DINO-WM.

    Input:
        x:       (B, T, N, D) per-patch embeddings for T frames
        act_emb: (B, T, D_act) per-frame action embeddings

    Output:
        (B, T, N, D_out) predicted patch embeddings

    Actions are tiled across patches and added to x before the transformer.
    The transformer uses standard Block (no AdaLN conditioning).
    """

    def __init__(
        self,
        *,
        num_frames,
        num_patches,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.num_frames = num_frames
        self.dropout = nn.Dropout(emb_dropout)

        # Decomposed positional embeddings: time + patch
        self.time_pos = nn.Parameter(torch.randn(1, num_frames, 1, input_dim))
        self.patch_pos = nn.Parameter(torch.randn(1, 1, num_patches, input_dim))

        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=Block,
        )

    def forward(self, x, act_emb):
        """
        x:       (B, T, N, D)
        act_emb: (B, T, D_act)
        returns: (B, T, N, D_out)
        """
        B, T, N, D = x.shape

        # Add decomposed positional embeddings
        x = x + self.time_pos[:, :T, :, :] + self.patch_pos[:, :, :N, :]

        # Add action embeddings: tile across patches then add
        act_tiled = act_emb.unsqueeze(2).expand(B, T, N, -1)  # (B, T, N, D_act)
        x = x + act_tiled

        # Flatten T*N into the sequence dim
        x = rearrange(x, "b t n d -> b (t n) d")
        x = self.dropout(x)

        # Standard transformer (no conditioning arg)
        x = self.transformer(x)

        # Unflatten back
        x = rearrange(x, "b (t n) d -> b t n d", t=T, n=N)
        return x
