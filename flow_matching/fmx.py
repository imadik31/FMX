"""Flux Matching loss utilities."""

from __future__ import annotations

import torch


def rbf_ovk_hessian(X: torch.Tensor, Y: torch.Tensor, sigma: float) -> torch.Tensor:
    """Compute operator-valued RBF kernel Hessian.

    Args:
        X: Tensor with shape ``[B, d]``.
        Y: Tensor with shape ``[B', d]``.
        sigma: RBF kernel width.

    Returns:
        A tensor with shape ``[B, B', d, d]`` containing the Hessian of the
        scalar RBF kernel ``k(x, y) = exp(-||x - y||^2 / (2 * sigma^2))``.
    """

    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("Inputs X and Y must be matrices with shape [B, d] and [B', d].")

    if X.size(-1) != Y.size(-1):
        raise ValueError("Input tensors must have the same feature dimension.")

    dif = X[:, None, :] - Y[None, :, :]  # [B, B', d]
    dist2 = (dif**2).sum(dim=-1)  # [B, B']
    denom = sigma**2 + 1e-12
    kxy = torch.exp(-0.5 * dist2 / denom)  # [B, B']

    outer = dif[..., :, None] * dif[..., None, :] / (denom**2)  # [B, B', d, d]
    eye = torch.eye(X.size(-1), device=X.device, dtype=X.dtype).view(1, 1, X.size(-1), X.size(-1))
    return (outer - eye / denom) * kxy[..., None, None]


def fmx_loss(
    Xm: torch.Tensor,
    vm: torch.Tensor,
    Xr: torch.Tensor,
    vr: torch.Tensor,
    sigma: float = 1.0,
) -> torch.Tensor:
    """Flux-MMD squared objective with an operator-valued RBF kernel.

    Args:
        Xm: Model particles with shape ``[B, d]``.
        vm: Model velocities with shape ``[B, d]``.
        Xr: Reference particles with shape ``[B_r, d]``.
        vr: Reference velocities with shape ``[B_r, d]``.
        sigma: RBF kernel width.

    Returns:
        Scalar tensor containing the FMX loss.
    """

    if Xm.shape != vm.shape:
        raise ValueError("Xm and vm must have the same shape.")

    if Xr.shape != vr.shape:
        raise ValueError("Xr and vr must have the same shape.")

    if Xm.size(-1) != Xr.size(-1):
        raise ValueError("Model and reference tensors must share the feature dimension.")

    Kmm = rbf_ovk_hessian(Xm, Xm, sigma)  # [B, B, d, d]
    Kmr = rbf_ovk_hessian(Xm, Xr, sigma)  # [B, B_r, d, d]

    vm_col = vm[:, None, :, None]  # [B, 1, d, 1]
    vm_row = vm[None, :, None, :]  # [1, B, 1, d]
    term_mm = (vm_col * Kmm * vm_row).sum(dim=(-1, -2)).mean()

    vr_row = vr[None, :, None, :]  # [1, B_r, 1, d]
    term_mr = (vm_col * Kmr * vr_row).sum(dim=(-1, -2)).mean()

    return term_mm - 2.0 * term_mr
