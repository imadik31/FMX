"""
Flux Matching (FMX) and Conditional Flux Matching (CFMX) loss implementations.

This module implements the Flux-MMD^2 loss with operator-valued kernel,
which matches the flux (joint distribution of positions and velocities)
between the model and reference distributions.

Two variants:
1. FMX: Model particles are pushed through the learned flow from x₀ to time t.
2. CFMX: Model and reference are evaluated at the same conditional bridge points
   x_t = (1-t)x₀ + tx₁, analogous to Conditional Flow Matching (CFM).

CFMX is generally more stable and easier to train, as it directly mirrors
the CFM approach but with flux matching instead of velocity regression.
"""

import torch


def median_heuristic(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Compute median heuristic for RBF kernel bandwidth.

    Args:
        X: Tensor of shape [B, d]
        Y: Tensor of shape [B', d]

    Returns:
        sigma: Recommended bandwidth parameter
    """
    # Subsample for efficiency if needed
    max_samples = 1000
    if X.size(0) > max_samples:
        idx_x = torch.randperm(X.size(0))[:max_samples]
        X = X[idx_x]
    if Y.size(0) > max_samples:
        idx_y = torch.randperm(Y.size(0))[:max_samples]
        Y = Y[idx_y]

    dif = X[:, None, :] - Y[None, :, :]
    dist = torch.sqrt((dif**2).sum(dim=-1) + 1e-12)
    median_dist = torch.median(dist)
    sigma = median_dist.item() / torch.sqrt(torch.tensor(2.0)).item()
    return max(sigma, 0.1)  # Avoid too small sigma


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
    sigma2 = sigma ** 2
    sigma4 = sigma ** 4

    dif = X[:, None, :] - Y[None, :, :]                    # [B,B',d]
    dist2 = (dif**2).sum(dim=-1)                           # [B,B']

    # Clamp distances to avoid numerical issues
    dist2 = torch.clamp(dist2, max=50.0 * sigma2)
    kxy = torch.exp(-0.5 * dist2 / sigma2)                 # [B,B']

    outer = dif[..., :, None] * dif[..., None, :] / sigma4  # [B,B',d,d]
    I = torch.eye(d, device=X.device, dtype=X.dtype).view(1, 1, d, d)
    Kxy = (outer - I / sigma2) * kxy[..., None, None]
    return Kxy  # [B,B',d,d]


def fmx_loss(
    Xm: torch.Tensor,
    vm: torch.Tensor,
    Xr: torch.Tensor,
    vr: torch.Tensor,
    sigma: float = 1.0,
    auto_sigma: bool = False,
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
        sigma: RBF kernel width (ignored if auto_sigma=True)
        auto_sigma: If True, compute sigma using median heuristic

    Returns:
        Scalar tensor loss
    """
    # Auto-compute sigma if requested
    if auto_sigma:
        with torch.no_grad():
            sigma = median_heuristic(Xm, Xr)

    Kmm = rbf_ovk_hessian(Xm, Xm, sigma)   # [B,B,d,d]
    Kmr = rbf_ovk_hessian(Xm, Xr, sigma)   # [B,Br,d,d]

    # (vm^T Kmm vm) using batched broadcasting
    vm_col = vm[:, None, :, None]          # [B,1,d,1]
    vm_row = vm[None, :, None, :]          # [1,B,1,d]
    term_mm = (vm_col * Kmm * vm_row).sum(dim=(-1, -2)).mean()

    vr_row = vr[None, :, None, :]          # [1,Br,1,d]
    term_mr = (vm_col * Kmr * vr_row).sum(dim=(-1, -2)).mean()

    # E[J*,J*] is constant wrt θ; drop it.
    loss = term_mm - 2.0 * term_mr

    # Clamp loss to avoid extreme values
    loss = torch.clamp(loss, min=-1e10, max=1e10)

    return loss
