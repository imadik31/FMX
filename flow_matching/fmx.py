"""
Flux Matching (FMX) and Conditional Flux Matching (CFMX) loss implementations.

This module implements the Flux-MMD^2 loss with operator-valued kernel,
which matches the flux (joint distribution of positions and velocities)
between the model and reference distributions.

Key insight:
- We minimize: obj = E[J,J] - 2*E[J,J*]  (constant E[J*,J*] dropped for efficiency)
- We monitor: mmd2 = E[J,J] - 2*E[J,J*] + E[J*,J*]  (full non-negative MMD^2)
- Both share the same gradients, but obj can be very negative while mmd2 is always >= 0
"""

import torch
from typing import Optional, Tuple


@torch.no_grad()
def median_heuristic_sigma(Xa: torch.Tensor, Xb: torch.Tensor, floor: float = 1e-2) -> float:
    """
    Median pairwise distance heuristic on a subsample; returns a *scalar* sigma.
    Uses Xa vs Xb cross-distances (more stable than within-batch if supports differ).

    Args:
        Xa: Tensor of shape [B, d]
        Xb: Tensor of shape [B', d]
        floor: Minimum sigma value to avoid numerical issues

    Returns:
        sigma: Recommended bandwidth parameter
    """
    B = min(Xa.shape[0], 2048)
    Ba = Xa[:B]
    Bb = Xb[:B]
    # Euclidean distances (no grad)
    d = torch.cdist(Ba, Bb, p=2)  # [B,B]
    med = torch.median(d)
    # floor to avoid sigma -> 0
    sigma = torch.clamp(med, min=floor).item()
    return sigma


def rbf_ovk_hessian(X: torch.Tensor, Y: torch.Tensor, sigma: float, ridge: float = 0.0):
    """
    Operator-valued kernel: K = ∇_x ∇_{x'}^T k_RBF(x, x'), k = exp(-||x-x'||^2 / (2 σ^2)).
    Returns Kxy: [B, B', d, d].
    Includes an optional tiny 'ridge' on the diagonal blocks for stability.

    Args:
        X: Tensor of shape [B, d]
        Y: Tensor of shape [B', d]
        sigma: RBF kernel bandwidth
        ridge: Ridge regularization added to diagonal blocks (if X == Y)

    Returns:
        Kxy: Tensor of shape [B, B', d, d]
    """
    B, d = X.shape
    dif = X[:, None, :] - Y[None, :, :]                    # [B,B',d]
    dist2 = (dif**2).sum(dim=-1)                           # [B,B']
    inv_sigma2 = 1.0 / (sigma**2 + 1e-12)
    inv_sigma4 = inv_sigma2 * inv_sigma2
    kxy = torch.exp(-0.5 * dist2 * inv_sigma2)             # [B,B']
    outer = dif[..., :, None] * dif[..., None, :] * inv_sigma4  # [B,B',d,d]
    I = torch.eye(d, device=X.device).view(1, 1, d, d)
    Kxy = (outer - I * inv_sigma2) * kxy[..., None, None]  # [B,B',d,d]

    if ridge > 0.0 and X.data_ptr() == Y.data_ptr():
        # add small ridge only on (i,i) blocks
        idx = torch.arange(B, device=X.device)
        Kxy[idx, idx, :, :] = Kxy[idx, idx, :, :] + ridge * I  # [B,d,d]
    return Kxy


def _quad_form(vm: torch.Tensor, K: torch.Tensor, vw: torch.Tensor) -> torch.Tensor:
    """
    Computes batched quadratic forms: mean_i,j vm_i^T K_ij vw_j.

    Args:
        vm: [B, d]
        K: [B, B', d, d]
        vw: [B', d]

    Returns:
        scalar tensor
    """
    vm_col = vm[:, None, :, None]     # [B,1,d,1]
    vw_row = vw[None, :, None, :]     # [1,B',1,d]
    return (vm_col * K * vw_row).sum(dim=(-1, -2)).mean()


def fmx_objective_and_metric(
    Xm: torch.Tensor,
    vm: torch.Tensor,
    Xr: torch.Tensor,
    vr: torch.Tensor,
    sigma: Optional[float] = None,
    use_auto_sigma: bool = False,
    ridge: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      obj  : E[J,J] - 2 E[J,J*]     (what you minimize; same grads as before)
      mmd2 : E[J,J] - 2 E[J,J*] + E[J*,J*]  (non-negative metric for logging)

    Both use the RBF OVK Hessian with stabilization.

    Args:
        Xm: Model particles, shape [B, d]
        vm: Model velocities, shape [B, d]
        Xr: Reference particles, shape [Br, d]
        vr: Reference velocities, shape [Br, d]
        sigma: RBF kernel width (ignored if use_auto_sigma=True)
        use_auto_sigma: If True, compute sigma using median heuristic
        ridge: Ridge regularization for diagonal blocks

    Returns:
        obj: Optimization objective (can be negative)
        mmd2: Full MMD^2 metric (always non-negative)
    """
    # Auto sigma (no grad)
    if use_auto_sigma or (sigma is None):
        with torch.no_grad():
            sigma_val = median_heuristic_sigma(Xm.detach(), Xr.detach())
            # scale a bit (empirically robust); you can tune this factor
            sigma_val = max(0.5 * sigma_val, 1e-2)
    else:
        sigma_val = float(sigma)

    Kmm = rbf_ovk_hessian(Xm, Xm, sigma=sigma_val, ridge=ridge)   # [B,B,d,d]
    Kmr = rbf_ovk_hessian(Xm, Xr, sigma=sigma_val, ridge=0.0)     # [B,Br,d,d]
    Krr = rbf_ovk_hessian(Xr, Xr, sigma=sigma_val, ridge=ridge)   # [Br,Br,d,d]

    term_mm = _quad_form(vm, Kmm, vm)     # E[J,J]
    term_mr = _quad_form(vm, Kmr, vr)     # E[J,J*]
    term_rr = _quad_form(vr, Krr, vr)     # E[J*,J*]  (const wrt θ)

    obj  = term_mm - 2.0 * term_mr        # what we minimize (same grads as before)
    mmd2 = obj + term_rr                   # non-negative metric to log
    return obj, mmd2
