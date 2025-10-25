"""
Flux Matching (FMX) loss implementation.

This module implements the Flux-MMD^2 loss with operator-valued kernel,
which matches the flux (joint distribution of positions and velocities)
between the model and reference distributions.
"""

import torch


def rbf_ovk_hessian(X: torch.Tensor, Y: torch.Tensor, sigma: float):
    """
    Operator-valued kernel K = ∇_x ∇_{x'}^T k_RBF(x, x'), with
      k(x,x') = exp(-||x-x'||^2 / (2 σ^2)).

    Args:
        X: Tensor of shape [B, d] representing particle positions
        Y: Tensor of shape [B', d] representing particle positions
        sigma: RBF kernel bandwidth parameter

    Returns:
        Kxy: Tensor of shape [B, B', d, d] - the operator-valued kernel matrix
    """
    B, d = X.shape
    dif = X[:, None, :] - Y[None, :, :]                    # [B,B',d]
    dist2 = (dif**2).sum(dim=-1)                           # [B,B']
    kxy = torch.exp(-0.5 * dist2 / (sigma**2 + 1e-12))     # [B,B']

    outer = dif[..., :, None] * dif[..., None, :] / (sigma**4 + 1e-12)  # [B,B',d,d]
    I = torch.eye(d, device=X.device).view(1, 1, d, d)
    Kxy = (outer - I / (sigma**2 + 1e-12)) * kxy[..., None, None]
    return Kxy  # [B,B',d,d]


def fmx_loss(
    Xm: torch.Tensor,
    vm: torch.Tensor,
    Xr: torch.Tensor,
    vr: torch.Tensor,
    sigma: float = 1.0,
) -> torch.Tensor:
    """
    Flux-MMD^2 with operator-valued kernel:
        L = E[J,J] - 2 E[J,J*] + const,
    where flux J is represented empirically by particle-velocity pairs.

    Args:
        Xm: Model particles, shape [B, d]
        vm: Model velocities, shape [B, d]
        Xr: Reference particles, shape [Br, d]
        vr: Reference velocities, shape [Br, d]
        sigma: RBF kernel width

    Returns:
        Scalar tensor loss
    """
    Kmm = rbf_ovk_hessian(Xm, Xm, sigma)   # [B,B,d,d]
    Kmr = rbf_ovk_hessian(Xm, Xr, sigma)   # [B,Br,d,d]

    # (vm^T Kmm vm) using batched broadcasting
    vm_col = vm[:, None, :, None]          # [B,1,d,1]
    vm_row = vm[None, :, None, :]          # [1,B,1,d]
    term_mm = (vm_col * Kmm * vm_row).sum(dim=(-1, -2)).mean()

    vr_row = vr[None, :, None, :]          # [1,Br,1,d]
    term_mr = (vm_col * Kmr * vr_row).sum(dim=(-1, -2)).mean()

    # E[J*,J*] is constant wrt θ; drop it.
    return term_mm - 2.0 * term_mr
