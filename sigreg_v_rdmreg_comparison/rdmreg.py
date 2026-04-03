import math
import torch
import torch.nn as nn


class RDMReg(nn.Module):
    """Rectified Distribution Matching Regularization.

    Pushes encoder features toward a Rectified Generalized Gaussian (RGG)
    distribution using the sliced 2-Wasserstein distance.

    Reference: https://yilunkuang.github.io/blog/2026/rectified-lp-jepa/

    Args:
        p:        Shape parameter of the underlying Generalized Gaussian.
                  p=1.0 → Rectified Laplace (recommended by paper ablations).
                  p=2.0 → Rectified Gaussian.
        mu:       Mean-shift of the underlying GN_p. mu=0 gives balanced
                  sparsity; lower values increase sparsity.
        num_proj: Number of random projection directions for the sliced
                  Wasserstein approximation. Paper default: 8192.
    """

    def __init__(self, p: float = 1.0, mu: float = 0.0, num_proj: int = 8192):
        super().__init__()
        self.p = p
        self.mu = mu
        self.num_proj = num_proj
        # sigma chosen so that GN_p(mu, sigma) has unit variance:
        #   sigma = Gamma(1/p)^(1/2) / (p^(1/p) * Gamma(3/p)^(1/2))
        sigma = math.gamma(1.0 / p) ** 0.5 / (
            p ** (1.0 / p) * math.gamma(3.0 / p) ** 0.5
        )
        self.sigma = sigma

    def _sample_rgg(self, shape: tuple, device: torch.device) -> torch.Tensor:
        """Sample from ReLU(GN_p(mu, sigma))."""
        if self.p == 1.0:
            noise = torch.distributions.Laplace(
                torch.tensor(self.mu, device=device),
                torch.tensor(self.sigma, device=device),
            ).sample(shape)
        elif self.p == 2.0:
            noise = torch.randn(shape, device=device) * self.sigma + self.mu
        else:
            # General case via Gamma + random sign
            # X ~ GN_p  ⟺  |X| ~ Gamma(1/p, p*sigma^p)^(1/p) · U{±1}
            g = torch.distributions.Gamma(
                torch.tensor(1.0 / self.p, device=device),
                torch.tensor(1.0 / (self.p * self.sigma ** self.p), device=device),
            ).sample(shape)
            magnitude = g ** (1.0 / self.p)
            sign = torch.randint(0, 2, shape, device=device).float() * 2 - 1
            noise = sign * magnitude + self.mu
        return noise.relu()

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """
        Compute RDMReg loss.

        Args:
            emb: Encoder embeddings of shape (B, T, D).

        Returns:
            Scalar sliced 2-Wasserstein loss.
        """
        B, T, D = emb.shape
        z = emb.reshape(B * T, D)  # (N, D) where N = B*T

        # Fresh RGG target samples (same N, D — needed to match sorted projections)
        y = self._sample_rgg((B * T, D), z.device)

        # Random unit projection vectors: (num_proj, D)
        proj = torch.randn(self.num_proj, D, device=z.device)
        proj = proj / proj.norm(dim=1, keepdim=True)

        # Project: (N, num_proj)
        pz = z @ proj.T
        py = y @ proj.T

        # Sort along batch/sample dimension
        pz_sorted, _ = torch.sort(pz, dim=0)
        py_sorted, _ = torch.sort(py, dim=0)

        return (pz_sorted - py_sorted).pow(2).mean()
