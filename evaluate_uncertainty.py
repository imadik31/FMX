"""
Evaluate uncertainty quality for Bayesian Flow Matching.

This script implements diagnostics to validate that the epistemic uncertainty
is meaningful and well-calibrated:

1. Uncertainty-error correlation: Does high uncertainty correlate with high error?
2. Coverage calibration: Do credible intervals contain the true values at the expected rate?
3. Posterior statistics: What are the learned posterior variances?

Important distinction:
- EPISTEMIC uncertainty: from weight posterior (Σ_epistemic)
- PREDICTIVE uncertainty: epistemic + aleatoric (Σ_pred = Σ_epistemic + σ_like² I)

For flow matching with deterministic velocity fields, use σ_like=0 (epistemic-only).
For regression with label noise, use σ_like > 0 (predictive coverage).

These metrics demonstrate the value of BFM over deterministic CFM.
"""

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import chi2, spearmanr
from torch import Tensor
from tqdm import tqdm

from flow_matching.datasets import TOY_DATASETS
from flow_matching.models.bayesian import BayesianMLP
from flow_matching.utils import set_seed


def joint_coverage_from_samples(
    V: Tensor,
    v_gt: Tensor,
    alphas: list,
    add_likelihood_sigma: float = None,
) -> dict:
    """
    Compute joint multivariate coverage using Mahalanobis distance.

    Tests if ground truth falls within α-credible ellipsoid defined by
    the posterior covariance. This is the correct way to evaluate
    calibration for vector-valued predictions.

    Args:
        V: Posterior samples [K, B, d]
        v_gt: Ground truth velocities [B, d]
        alphas: List of credible levels (e.g., [0.5, 0.8, 0.95])
        add_likelihood_sigma: If not None, add σ²I to covariance (predictive coverage)

    Returns:
        dict mapping alpha -> observed coverage fraction
    """
    K, B, d = V.shape

    # Compute posterior mean and covariance for each sample
    mu = V.mean(dim=0)  # [B, d]

    # Centered samples
    V_centered = V - mu.unsqueeze(0)  # [K, B, d]

    # Sample covariance per point: [B, d, d]
    # Sigma[i] = (1/(K-1)) * sum_k (V[k,i] - mu[i]) (V[k,i] - mu[i])^T
    Sigma = torch.einsum('kbd,kbe->bde', V_centered, V_centered) / (K - 1 + 1e-9)

    # Add likelihood noise if requested (predictive coverage)
    if add_likelihood_sigma is not None:
        eye = torch.eye(d, device=Sigma.device).unsqueeze(0)  # [1, d, d]
        Sigma = Sigma + (add_likelihood_sigma ** 2) * eye

    # Regularize and invert covariances
    eps = 1e-6
    eye = torch.eye(d, device=Sigma.device).unsqueeze(0)
    Sigma_inv = torch.linalg.inv(Sigma + eps * eye)  # [B, d, d]

    # Compute Mahalanobis distance: Δ_i = (v_gt[i] - mu[i])^T Sigma_inv[i] (v_gt[i] - mu[i])
    diff = (v_gt - mu).unsqueeze(-1)  # [B, d, 1]
    mah2 = torch.einsum('bdi,bij,bjk->bk', diff.transpose(1, 2), Sigma_inv, diff)
    mah2 = mah2.squeeze(-1).squeeze(-1)  # [B]

    # Test coverage for each alpha
    coverage = {}
    for alpha in alphas:
        # Chi-squared threshold for α-credible ellipsoid
        thresh = chi2.ppf(alpha, df=d)

        # Count points inside ellipsoid
        inside = (mah2 <= thresh).float()
        observed = inside.mean().item()
        coverage[alpha] = observed

    return coverage


def marginal_coverage_from_samples(
    V: Tensor,
    v_gt: Tensor,
    alphas: list,
    add_likelihood_sigma: Optional[float] = None
) -> dict:
    """
    Compute marginal (per-dimension) coverage using quantile intervals.

    This ignores cross-dimension correlations and will be anti-conservative
    (under-report coverage). Kept for comparison with joint coverage.

    Args:
        V: Posterior samples [K, B, d]
        v_gt: Ground truth velocities [B, d]
        alphas: List of credible levels
        add_likelihood_sigma: If provided, inflate variance by sigma_like^2 for predictive coverage

    Returns:
        dict mapping alpha -> observed marginal coverage
    """
    K, B, d = V.shape

    # If adding likelihood noise for predictive coverage
    if add_likelihood_sigma is not None:
        # For marginal coverage with Gaussian likelihood, we sample noise for each posterior sample
        # V_pred = V_epistemic + N(0, sigma_like^2)
        V = V + add_likelihood_sigma * torch.randn_like(V)

    coverage = {}

    for alpha in alphas:
        # Compute quantile intervals per dimension
        lower_q = (1 - alpha) / 2
        upper_q = 1 - lower_q

        lower = torch.quantile(V, lower_q, dim=0)  # [B, d]
        upper = torch.quantile(V, upper_q, dim=0)  # [B, d]

        # Check if ground truth is within interval (per dimension)
        inside = (v_gt >= lower) & (v_gt <= upper)  # [B, d]

        # Average over samples and dimensions
        observed = inside.float().mean().item()
        coverage[alpha] = observed

    return coverage


def compute_uncertainty_error_correlation(
    model: BayesianMLP,
    dataset,
    n_samples: int = 8192,
    n_posterior: int = 50,
    device: torch.device = None,
) -> dict:
    """
    Compute correlation between epistemic uncertainty and prediction error.

    High Spearman correlation indicates that the model's uncertainty is
    informative: it knows what it doesn't know.

    Args:
        model: Trained Bayesian flow model
        dataset: Dataset to sample from
        n_samples: Number of test samples
        n_posterior: Number of posterior samples for uncertainty

    Returns:
        dict with metrics: rho (Spearman), uncertainties, errors, etc.
    """
    if device is None:
        device = next(model.parameters()).device

    # Sample test data
    print(f"Sampling {n_samples} test points...")
    x_1 = dataset.sample(n_samples)
    x_0 = torch.randn_like(x_1).to(device)
    t = torch.rand(x_1.size(0), 1).to(device)

    # Compute probability path and ground truth
    x_t = (1 - t) * x_0 + t * x_1
    v_gt = x_1 - x_0

    # Collect posterior samples
    print(f"Collecting {n_posterior} posterior samples...")
    preds = []
    with torch.no_grad():
        for _ in tqdm(range(n_posterior), desc="Posterior samples"):
            v_pred = model.forward_sample(x_t, t, n_mc=1).squeeze(0)  # [B, d]
            preds.append(v_pred)

    # Stack predictions: [K, B, d]
    V = torch.stack(preds, dim=0)

    # Compute statistics
    mu = V.mean(dim=0)  # [B, d] - posterior mean
    std = V.std(dim=0)  # [B, d] - epistemic uncertainty

    # Compute uncertainty magnitude (L2 norm)
    uncertainty = std.norm(dim=-1)  # [B]

    # Compute error magnitude (RMSE)
    error = (mu - v_gt).pow(2).sum(dim=-1).sqrt()  # [B]

    # Move to CPU for stats
    uncertainty_cpu = uncertainty.cpu().numpy()
    error_cpu = error.cpu().numpy()

    # Compute Spearman correlation
    rho, p_value = spearmanr(uncertainty_cpu, error_cpu)

    print(f"\n" + "=" * 80)
    print("Uncertainty-Error Correlation")
    print("=" * 80)
    print(f"Spearman ρ: {rho:.4f} (p-value: {p_value:.2e})")
    print(f"Mean uncertainty: {uncertainty_cpu.mean():.6f}")
    print(f"Mean error (RMSE): {error_cpu.mean():.6f}")
    print(f"Std uncertainty: {uncertainty_cpu.std():.6f}")
    print(f"Std error: {error_cpu.std():.6f}")
    print("=" * 80)

    if rho > 0.3:
        print("✓ Good correlation: Uncertainty is informative!")
    elif rho > 0.1:
        print("⚠ Moderate correlation: Some informativeness")
    else:
        print("✗ Low correlation: Uncertainty may not be well-calibrated")

    return {
        'rho': rho,
        'p_value': p_value,
        'uncertainty': uncertainty_cpu,
        'error': error_cpu,
        'mu': mu.cpu(),
        'std': std.cpu(),
        'v_gt': v_gt.cpu(),
    }


def compute_coverage_calibration(
    model: BayesianMLP,
    dataset,
    n_samples: int = 2048,
    n_posterior: int = 100,
    alphas: list = None,
    sigma_likelihood: float = 0.0,
    device: torch.device = None,
) -> dict:
    """
    Compute coverage calibration for credible intervals.

    A well-calibrated model's α-credible intervals should contain the
    true value approximately α fraction of the time.

    Args:
        model: Trained Bayesian flow model
        dataset: Dataset to sample from
        n_samples: Number of test samples
        n_posterior: Number of posterior samples
        alphas: List of credible interval levels (e.g., [0.5, 0.8, 0.95])
        sigma_likelihood: Aleatoric noise std for predictive coverage.
                         Use 0.0 for epistemic-only coverage (deterministic velocity fields)

    Returns:
        dict with coverage statistics
    """
    if device is None:
        device = next(model.parameters()).device

    if alphas is None:
        alphas = [0.5, 0.68, 0.8, 0.9, 0.95]

    # Sample test data
    print(f"\nSampling {n_samples} test points for calibration...")
    x_1 = dataset.sample(n_samples)
    x_0 = torch.randn_like(x_1).to(device)

    # Test at multiple time points
    time_points = [0.25, 0.5, 0.75]
    all_coverages_joint = {alpha: [] for alpha in alphas}
    all_coverages_marginal = {alpha: [] for alpha in alphas}

    for t_val in time_points:
        print(f"\nTesting at t={t_val}...")
        t = torch.full((n_samples, 1), t_val, device=device)

        # Compute ground truth
        x_t = (1 - t) * x_0 + t * x_1
        v_gt = x_1 - x_0

        # Collect posterior samples
        preds = []
        with torch.no_grad():
            for _ in tqdm(range(n_posterior), desc=f"t={t_val}"):
                v_pred = model.forward_sample(x_t, t, n_mc=1).squeeze(0)
                preds.append(v_pred)

        # Stack: [K, B, d]
        V = torch.stack(preds, dim=0).cpu()
        v_gt = v_gt.cpu()

        # Compute JOINT multivariate coverage (correct for d>1)
        # Using PREDICTIVE coverage: Σ_pred = Σ_epistemic + σ_like² I
        cover_joint = joint_coverage_from_samples(V, v_gt, alphas, add_likelihood_sigma=sigma_likelihood)
        for alpha in alphas:
            all_coverages_joint[alpha].append(cover_joint[alpha])

        # Compute MARGINAL per-dimension coverage (for comparison)
        cover_marginal = marginal_coverage_from_samples(V, v_gt, alphas, add_likelihood_sigma=sigma_likelihood)
        for alpha in alphas:
            all_coverages_marginal[alpha].append(cover_marginal[alpha])

    # Average coverage across time points
    avg_coverages_joint = {alpha: np.mean(covs) for alpha, covs in all_coverages_joint.items()}
    avg_coverages_marginal = {alpha: np.mean(covs) for alpha, covs in all_coverages_marginal.items()}

    print(f"\n" + "=" * 80)
    print("Coverage Calibration - JOINT (Multivariate Ellipsoid)")
    print("=" * 80)
    print("This is the CORRECT test for vector-valued predictions.")
    if sigma_likelihood > 0:
        print(f"Testing PREDICTIVE coverage: Σ_pred = Σ_epistemic + σ_like² I (σ_like={sigma_likelihood:.4f})")
    else:
        print("Testing EPISTEMIC coverage: Σ_pred = Σ_epistemic (appropriate for deterministic fields)")
    print(f"{'Alpha':>6} | {'Expected':>8} | {'Observed':>8} | {'Diff':>7}")
    print("-" * 80)

    for alpha in alphas:
        expected = alpha
        observed = avg_coverages_joint[alpha]
        diff = observed - expected
        status = "✓" if abs(diff) < 0.05 else "⚠" if abs(diff) < 0.1 else "✗"
        print(f"{status} {alpha:>4.2f} | {expected:>8.3f} | {observed:>8.3f} | {diff:>+7.3f}")

    print("=" * 80)

    print(f"\n" + "=" * 80)
    print("Coverage Calibration - MARGINAL (Per-Dimension)")
    print("=" * 80)
    print("This ignores correlations and will under-report coverage (for comparison).")
    print(f"{'Alpha':>6} | {'Expected':>8} | {'Observed':>8} | {'Diff':>7}")
    print("-" * 80)

    for alpha in alphas:
        expected = alpha
        observed = avg_coverages_marginal[alpha]
        diff = observed - expected
        status = "✓" if abs(diff) < 0.05 else "⚠" if abs(diff) < 0.1 else "✗"
        print(f"{status} {alpha:>4.2f} | {expected:>8.3f} | {observed:>8.3f} | {diff:>+7.3f}")

    print("=" * 80)

    return {
        'alphas': alphas,
        'coverages_joint': avg_coverages_joint,
        'coverages_marginal': avg_coverages_marginal,
        'all_coverages_joint': all_coverages_joint,
        'all_coverages_marginal': all_coverages_marginal,
    }


def analyze_posterior_statistics(model: BayesianMLP) -> dict:
    """
    Analyze learned posterior statistics (mean and std of weights).

    Returns:
        dict with posterior statistics
    """
    with torch.no_grad():
        W_mu = model.head.W_mu.cpu()
        b_mu = model.head.b_mu.cpu()
        W_std = torch.exp(model.head.W_logsig).cpu()
        b_std = torch.exp(model.head.b_logsig).cpu()

    print(f"\n" + "=" * 80)
    print("Posterior Statistics")
    print("=" * 80)
    print(f"Weight (W) posterior:")
    print(f"  Mean magnitude: {W_mu.abs().mean().item():.6f}")
    print(f"  Mean std: {W_std.mean().item():.6f}")
    print(f"  Max std: {W_std.max().item():.6f}")
    print(f"  Min std: {W_std.min().item():.6f}")
    print(f"\nBias (b) posterior:")
    print(f"  Mean magnitude: {b_mu.abs().mean().item():.6f}")
    print(f"  Mean std: {b_std.mean().item():.6f}")
    print(f"  Max std: {b_std.max().item():.6f}")
    print(f"  Min std: {b_std.min().item():.6f}")
    print("=" * 80)

    return {
        'W_mu_mean': W_mu.abs().mean().item(),
        'W_std_mean': W_std.mean().item(),
        'b_mu_mean': b_mu.abs().mean().item(),
        'b_std_mean': b_std.mean().item(),
    }


def plot_uncertainty_error_scatter(results: dict, output_path: Path):
    """Plot scatter of uncertainty vs error."""
    uncertainty = results['uncertainty']
    error = results['error']
    rho = results['rho']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter plot
    ax = axes[0]
    ax.scatter(uncertainty, error, alpha=0.3, s=10)
    ax.set_xlabel("Epistemic Uncertainty (L2 norm)", fontsize=12)
    ax.set_ylabel("Prediction Error (RMSE)", fontsize=12)
    ax.set_title(f"Uncertainty vs Error (ρ={rho:.3f})", fontsize=14)
    ax.grid(True, alpha=0.3)

    # Add trend line
    z = np.polyfit(uncertainty, error, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(uncertainty.min(), uncertainty.max(), 100)
    ax.plot(x_trend, p(x_trend), "r--", linewidth=2, alpha=0.8, label=f"y={z[0]:.2f}x+{z[1]:.2f}")
    ax.legend()

    # Binned statistics
    ax = axes[1]
    n_bins = 20
    bins = np.linspace(uncertainty.min(), uncertainty.max(), n_bins + 1)
    bin_indices = np.digitize(uncertainty, bins)

    bin_centers = []
    bin_mean_errors = []
    bin_std_errors = []

    for i in range(1, n_bins + 1):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_centers.append(bins[i - 1] + (bins[i] - bins[i - 1]) / 2)
            bin_mean_errors.append(error[mask].mean())
            bin_std_errors.append(error[mask].std())

    ax.errorbar(bin_centers, bin_mean_errors, yerr=bin_std_errors, fmt='o-', capsize=5, linewidth=2)
    ax.set_xlabel("Epistemic Uncertainty (binned)", fontsize=12)
    ax.set_ylabel("Mean Prediction Error", fontsize=12)
    ax.set_title("Binned Uncertainty vs Error", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nScatter plot saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=TOY_DATASETS.keys(), required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/uncertainty_eval")
    parser.add_argument("--n-samples", type=int, default=8192)
    parser.add_argument("--n-posterior", type=int, default=50)
    parser.add_argument("--n-calibration-samples", type=int, default=2048)
    parser.add_argument("--n-calibration-posterior", type=int, default=100)
    parser.add_argument("--sigma-like", type=float, default=None,
                        help="Override sigma_likelihood for predictive coverage. "
                             "If None, use checkpoint config 'sigma_likelihood' if present; else 0.0 "
                             "(0.0 = epistemic-only coverage for deterministic velocity fields)")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    output_dir = Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Bayesian Flow Matching - Uncertainty Evaluation")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Checkpoint: {args.checkpoint}")
    print("=" * 80)

    # Load dataset
    dataset = TOY_DATASETS[args.dataset](device=device)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)

    if isinstance(checkpoint, dict) and 'config' in checkpoint:
        config = checkpoint['config']
        state_dict = checkpoint['state_dict']
    else:
        # Infer architecture
        state_dict = checkpoint
        backbone_keys = [k for k in state_dict.keys() if k.startswith('backbone.') and 'weight' in k]
        max_idx = max([int(k.split('.')[1]) for k in backbone_keys])
        inferred_num_layers = (max_idx + 1) // 2
        inferred_hidden_dim = state_dict['backbone.0.weight'].shape[0]

        config = {
            'dim': dataset.dim,
            'time_dim': 1,
            'hidden_dim': inferred_hidden_dim,
            'num_layers': inferred_num_layers,
            'sigma_p': 1.0,
        }

    # Add defaults for backward compatibility
    config.setdefault('init_sigma_likelihood', 1.0)
    config.setdefault('dropout_p', 0.1)

    # Create model
    model = BayesianMLP(
        dim=config['dim'],
        time_dim=config['time_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        sigma_p=config['sigma_p'],
        init_sigma_likelihood=config['init_sigma_likelihood'],
        dropout_p=config['dropout_p'],
    ).to(device)

    model.load_state_dict(state_dict)
    print(f"Loaded model: hidden_dim={config['hidden_dim']}, num_layers={config['num_layers']}")

    # 1. Analyze posterior statistics
    posterior_stats = analyze_posterior_statistics(model)

    # 2. Uncertainty-error correlation
    print(f"\n" + "=" * 80)
    print("Computing uncertainty-error correlation...")
    print("=" * 80)
    correlation_results = compute_uncertainty_error_correlation(
        model, dataset, n_samples=args.n_samples, n_posterior=args.n_posterior, device=device
    )

    # Plot scatter
    plot_uncertainty_error_scatter(
        correlation_results,
        output_path=output_dir / "uncertainty_error_scatter.png"
    )

    # 3. Coverage calibration
    print(f"\n" + "=" * 80)
    print("Computing coverage calibration...")
    print("=" * 80)

    # Determine sigma_likelihood for predictive coverage
    # Use CLI override if provided, else checkpoint config, else 0.0 (epistemic-only)
    sigma_like_for_coverage = args.sigma_like
    if sigma_like_for_coverage is None:
        sigma_like_for_coverage = config.get('sigma_likelihood', 0.0)

    print(f"Using σ_likelihood = {sigma_like_for_coverage:.4f} for predictive coverage")
    if sigma_like_for_coverage == 0.0:
        print("  (0.0 = epistemic-only coverage, appropriate for deterministic velocity fields)")

    calibration_results = compute_coverage_calibration(
        model,
        dataset,
        n_samples=args.n_calibration_samples,
        n_posterior=args.n_calibration_posterior,
        sigma_likelihood=sigma_like_for_coverage,
        device=device
    )

    # Save results
    results = {
        'posterior_stats': posterior_stats,
        'correlation': {
            'rho': correlation_results['rho'],
            'p_value': correlation_results['p_value'],
        },
        'calibration_joint': calibration_results['coverages_joint'],
        'calibration_marginal': calibration_results['coverages_marginal'],
    }

    import json
    with open(output_dir / "uncertainty_metrics.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics saved to {output_dir / 'uncertainty_metrics.json'}")

    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print(f"Results saved to {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
