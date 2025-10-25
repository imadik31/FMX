"""
Fast Flux Matching (FMX) using curl-free Random Fourier Features (RFF).

Replaces expensive O(B²) operator-valued kernels with O(BD) feature embeddings,
making FMX practical for large batch sizes while preserving the curl-free RKHS theory.

Key idea:
- For RBF kernel k(x,x'), sample w_ℓ ~ N(0, σ⁻² I)
- Curl-free vector features: g_ℓ(x) = ∇φ_ℓ(x) where φ_ℓ(x) = cos/sin(w_ℓ^T x)
- Flux embedding: μ_J ≈ (1/B) Σ [G(x_i)^T v(x_i)] ∈ R^{2D}
- Loss: ||μ_{J_θ} - μ_{J*}||²

Complexity: O(BDd) vs O(B²d) for exact kernel - linear in batch size!
"""

import torch
from typing import Tuple


class CurlFreeRFF(torch.nn.Module):
    """
    Curl-free Random Fourier Features for fast flux matching.

    Args:
        d: Input dimension
        D: Number of random features (typically 128-512)
        sigma: RBF kernel bandwidth
    """

    def __init__(self, d: int, D: int, sigma: float):
        super().__init__()
        self.d = d
        self.D = D
        # Sample random frequencies: W ~ N(0, σ⁻² I)
        W = torch.randn(D, d) / (sigma + 1e-12)
        self.register_buffer("W", W)

    def update_sigma(self, sigma: float):
        """Re-sample random features with new bandwidth."""
        W = torch.randn(self.D, self.d, device=self.W.device, dtype=self.W.dtype) / (sigma + 1e-12)
        self.W.copy_(W)

    def embed_flux(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Embed flux J = (X, V) into curl-free RKHS feature space.

        μ_J ≈ (1/B) Σ_i [√(2/D) * ([cos(z)*W]·v, [-sin(z)*W]·v)]

        Args:
            x: Positions [B, d]
            v: Velocities [B, d]

        Returns:
            mu: Flux embedding [2D]
        """
        B = x.size(0)
        z = x @ self.W.t()                          # [B, D]
        c, s = torch.cos(z), torch.sin(z)           # [B, D], [B, D]

        # Compute v·W^T once
        vW = v @ self.W.t()                         # [B, D]

        # Two blocks: cos*(v·w) and -sin*(v·w)
        # These correspond to gradients of cos/sin features
        block1 = (c * vW).sum(dim=0)                # [D]
        block2 = (-s * vW).sum(dim=0)               # [D]

        mu = torch.cat([block1, block2], dim=0)     # [2D]
        mu = ((2.0 / self.D) ** 0.5) * (mu / B)     # normalize
        return mu


def fast_fmx_loss(
    rff: CurlFreeRFF,
    Xm: torch.Tensor,
    vm: torch.Tensor,
    Xr: torch.Tensor,
    vr: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fast FMX loss using RFF embeddings.

    Returns both objective (for optimization) and squared distance (for logging).
    With RFF, they're the same since we don't drop the constant term.

    Args:
        rff: CurlFreeRFF module with random features
        Xm: Model positions [B, d]
        vm: Model velocities [B, d]
        Xr: Reference positions [B, d]
        vr: Reference velocities [B, d]

    Returns:
        loss: ||μ_{J_θ} - μ_{J*}||²
        mmd2: Same as loss (no separate constant term with RFF)
    """
    mu_m = rff.embed_flux(Xm, vm)  # [2D]
    mu_r = rff.embed_flux(Xr, vr)  # [2D]

    loss = ((mu_m - mu_r) ** 2).mean()
    return loss, loss  # Both are the same for RFF
