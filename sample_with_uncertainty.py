"""
Sample from Bayesian Flow Matching with uncertainty quantification.

This script demonstrates:
1. Sampling with multiple posterior draws for epistemic uncertainty
2. Visualizing uncertainty in trajectories and final samples
3. Calibration analysis for uncertainty estimates
"""

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Ellipse
from torch import Tensor
from tqdm import tqdm

from flow_matching.datasets import TOY_DATASETS
from flow_matching.models.bayesian import BayesianMLP
from flow_matching.solver import ODESolver, ModelWrapper
from flow_matching.utils import set_seed


class BayesianODEWrapper(ModelWrapper):
    """Wrapper that samples from posterior during ODE integration."""

    def __init__(self, model, use_mean=False):
        super().__init__(model)
        self.use_mean = use_mean

    def forward(self, x: Tensor, t: Tensor, **extras) -> Tensor:
        if self.use_mean:
            return self.model.forward_mean(x_t=x, t=t)
        else:
            # Single MC sample
            return self.model.forward_sample(x_t=x, t=t, n_mc=1).squeeze(0)


def sample_with_uncertainty(
    model: BayesianMLP,
    x_init: Tensor,
    n_posterior_samples: int = 10,
    step_size: float = 0.01,
    method: str = "euler",
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Sample from flow with uncertainty quantification.

    Args:
        model: Trained Bayesian flow model
        x_init: Initial noise samples [batch_size, dim]
        n_posterior_samples: Number of posterior samples (K)
        step_size: ODE solver step size
        method: ODE solver method

    Returns:
        mean: Mean prediction [batch_size, dim]
        std: Standard deviation (epistemic uncertainty) [batch_size, dim]
        samples: All posterior samples [n_posterior_samples, batch_size, dim]
    """
    all_samples = []

    for _ in range(n_posterior_samples):
        # Create wrapper that samples from posterior
        wrapped = BayesianODEWrapper(model, use_mean=False)
        solver = ODESolver(wrapped)

        # Integrate ODE
        x_final = solver.sample(
            x_init=x_init,
            step_size=step_size,
            method=method,
            time_grid=torch.linspace(0, 1, 100, device=x_init.device),
        )

        all_samples.append(x_final)

    # Stack samples [K, B, d]
    samples = torch.stack(all_samples, dim=0)

    # Compute statistics
    mean = samples.mean(dim=0)
    std = samples.std(dim=0)

    return mean, std, samples


def visualize_uncertainty_samples(
    dataset,
    model: BayesianMLP,
    n_samples: int = 500,
    n_posterior: int = 10,
    output_path: Path = None,
):
    """Visualize samples with uncertainty ellipses."""
    device = next(model.parameters()).device

    # Generate samples with uncertainty
    x_0 = torch.randn(n_samples, dataset.dim, device=device)

    print(f"Generating {n_samples} samples with {n_posterior} posterior draws...")
    mean, std, all_samples = sample_with_uncertainty(
        model, x_0, n_posterior_samples=n_posterior, step_size=0.01, method="euler"
    )

    # Move to CPU for plotting
    mean = mean.cpu().numpy()
    std = std.cpu().numpy()
    all_samples = all_samples.cpu().numpy()

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: True data distribution
    ax = axes[0]
    true_samples = dataset.sample(5000).cpu().numpy()
    ax.scatter(true_samples[:, 0], true_samples[:, 1], alpha=0.3, s=10, c="#1f77b4")
    ax.set_title("True Data Distribution", fontsize=14)
    ax.set_xlabel("$x_1$", fontsize=12)
    ax.set_ylabel("$x_2$", fontsize=12)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Plot 2: Posterior mean samples
    ax = axes[1]
    ax.scatter(mean[:, 0], mean[:, 1], alpha=0.5, s=10, c="#ff7f0e")
    ax.set_title("Posterior Mean Samples", fontsize=14)
    ax.set_xlabel("$x_1$", fontsize=12)
    ax.set_ylabel("$x_2$", fontsize=12)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Plot 3: Samples with uncertainty ellipses
    ax = axes[2]

    # Plot a subset with uncertainty ellipses
    n_show = 100
    indices = np.random.choice(n_samples, n_show, replace=False)

    for idx in indices:
        # Plot all posterior samples for this point
        ax.scatter(
            all_samples[:, idx, 0],
            all_samples[:, idx, 1],
            alpha=0.2,
            s=5,
            c="#2ca02c",
        )

        # Draw uncertainty ellipse (2 std)
        if std[idx, 0] > 1e-6 and std[idx, 1] > 1e-6:
            ellipse = Ellipse(
                xy=(mean[idx, 0], mean[idx, 1]),
                width=2 * std[idx, 0],
                height=2 * std[idx, 1],
                edgecolor="#d62728",
                facecolor="none",
                linewidth=0.5,
                alpha=0.5,
            )
            ax.add_patch(ellipse)

    # Plot mean
    ax.scatter(mean[indices, 0], mean[indices, 1], alpha=0.8, s=15, c="#ff7f0e", zorder=10)

    ax.set_title(f"Samples with Epistemic Uncertainty (K={n_posterior})", fontsize=14)
    ax.set_xlabel("$x_1$", fontsize=12)
    ax.set_ylabel("$x_2$", fontsize=12)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Uncertainty visualization saved to {output_path}")
    else:
        plt.show()

    plt.close()

    # Print uncertainty statistics
    mean_std = std.mean(axis=0)
    print(f"\nUncertainty Statistics:")
    print(f"  Mean epistemic uncertainty (dim 1): {mean_std[0]:.6f}")
    print(f"  Mean epistemic uncertainty (dim 2): {mean_std[1]:.6f}")
    print(f"  Max epistemic uncertainty (dim 1): {std[:, 0].max():.6f}")
    print(f"  Max epistemic uncertainty (dim 2): {std[:, 1].max():.6f}")


def analyze_trajectory_uncertainty(
    dataset,
    model: BayesianMLP,
    n_trajectories: int = 5,
    n_posterior: int = 20,
    output_path: Path = None,
):
    """Analyze and visualize uncertainty along trajectories."""
    device = next(model.parameters()).device

    # Initial points
    x_0 = torch.randn(n_trajectories, dataset.dim, device=device)

    # Time grid
    time_grid = torch.linspace(0, 1, 50, device=device)

    print(f"Analyzing {n_trajectories} trajectories with {n_posterior} posterior draws...")

    # Collect trajectories for each posterior sample
    all_trajectories = []  # [K, n_traj, n_steps, dim]

    for k in tqdm(range(n_posterior), desc="Posterior samples"):
        wrapped = BayesianODEWrapper(model, use_mean=False)
        solver = ODESolver(wrapped)

        traj = solver.sample(
            x_init=x_0,
            step_size=None,  # Use adaptive
            method="dopri5",
            time_grid=time_grid,
            return_intermediates=True,
        )

        # traj is already a tensor [n_steps, n_traj, dim]
        all_trajectories.append(traj)

    # Stack: [K, n_steps, n_traj, dim]
    all_trajectories = torch.stack(all_trajectories, dim=0)

    # Compute mean and std across posterior samples
    # [n_steps, n_traj, dim]
    mean_traj = all_trajectories.mean(dim=0)
    std_traj = all_trajectories.std(dim=0)

    # Move to CPU
    mean_traj = mean_traj.cpu().numpy()
    std_traj = std_traj.cpu().numpy()
    time_grid = time_grid.cpu().numpy()

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Trajectories with uncertainty bands
    ax = axes[0]

    for i in range(n_trajectories):
        # Mean trajectory
        ax.plot(
            mean_traj[:, i, 0],
            mean_traj[:, i, 1],
            linewidth=2,
            alpha=0.8,
            label=f"Traj {i+1}",
        )

        # Uncertainty ellipses at selected time points
        for t_idx in range(0, len(time_grid), 10):
            if std_traj[t_idx, i, 0] > 1e-6 and std_traj[t_idx, i, 1] > 1e-6:
                ellipse = Ellipse(
                    xy=(mean_traj[t_idx, i, 0], mean_traj[t_idx, i, 1]),
                    width=2 * std_traj[t_idx, i, 0],
                    height=2 * std_traj[t_idx, i, 1],
                    edgecolor="gray",
                    facecolor="none",
                    linewidth=0.5,
                    alpha=0.3,
                )
                ax.add_patch(ellipse)

    ax.set_title("Trajectories with Epistemic Uncertainty", fontsize=14)
    ax.set_xlabel("$x_1$", fontsize=12)
    ax.set_ylabel("$x_2$", fontsize=12)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Plot 2: Uncertainty vs time
    ax = axes[1]

    for i in range(n_trajectories):
        # Total uncertainty (Frobenius norm)
        total_std = np.sqrt(std_traj[:, i, 0] ** 2 + std_traj[:, i, 1] ** 2)
        ax.plot(time_grid, total_std, linewidth=2, alpha=0.8, label=f"Traj {i+1}")

    ax.set_title("Uncertainty Evolution over Time", fontsize=14)
    ax.set_xlabel("Time $t$", fontsize=12)
    ax.set_ylabel("Epistemic Uncertainty (L2 norm)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Trajectory uncertainty visualization saved to {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=TOY_DATASETS.keys(), required=True)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained BFM model")
    parser.add_argument("--output-dir", type=str, default="outputs/uncertainty")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--n-posterior", type=int, default=20, help="Number of posterior draws")
    parser.add_argument("--n-trajectories", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--sigma-prior", type=float, default=1.0)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    output_dir = Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Bayesian Flow Matching - Uncertainty Quantification")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Posterior samples: {args.n_posterior}")
    print("=" * 80)

    # Load dataset
    dataset = TOY_DATASETS[args.dataset](device=device)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Check if checkpoint contains config (new format) or just state_dict (old format)
    if isinstance(checkpoint, dict) and 'config' in checkpoint:
        # New format with config
        config = checkpoint['config']
        state_dict = checkpoint['state_dict']
        print(f"Loaded config from checkpoint:")
        print(f"  dim={config['dim']}, hidden_dim={config['hidden_dim']}, "
              f"num_layers={config['num_layers']}, sigma_p={config['sigma_p']}")
    else:
        # Old format (just state_dict) - infer architecture from state_dict keys
        state_dict = checkpoint

        # Infer num_layers from backbone structure
        # Backbone has structure: [Linear, SiLU] * num_layers
        backbone_keys = [k for k in state_dict.keys() if k.startswith('backbone.') and 'weight' in k]
        if backbone_keys:
            max_idx = max([int(k.split('.')[1]) for k in backbone_keys])
            inferred_num_layers = (max_idx + 1) // 2
        else:
            inferred_num_layers = args.num_layers

        # Infer hidden_dim from first backbone layer
        if 'backbone.0.weight' in state_dict:
            inferred_hidden_dim = state_dict['backbone.0.weight'].shape[0]
        else:
            inferred_hidden_dim = args.hidden_dim

        config = {
            'dim': dataset.dim,
            'time_dim': 1,
            'hidden_dim': inferred_hidden_dim,
            'num_layers': inferred_num_layers,
            'sigma_p': args.sigma_prior,
        }

        print(f"Warning: Old checkpoint format detected. Inferred architecture from state_dict:")
        print(f"  dim={config['dim']}, hidden_dim={config['hidden_dim']}, "
              f"num_layers={config['num_layers']}, sigma_p={config['sigma_p']}")

    # Add defaults for backward compatibility
    config.setdefault('init_sigma_likelihood', 1.0)
    config.setdefault('dropout_p', 0.1)

    # Create model with loaded/inferred config
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
    print(f"Loaded model from {args.checkpoint}")

    # Visualize uncertainty in samples
    print("\n" + "=" * 80)
    print("Visualizing uncertainty in samples...")
    print("=" * 80)
    visualize_uncertainty_samples(
        dataset,
        model,
        n_samples=args.n_samples,
        n_posterior=args.n_posterior,
        output_path=output_dir / "uncertainty_samples.png",
    )

    # Analyze trajectory uncertainty
    print("\n" + "=" * 80)
    print("Analyzing trajectory uncertainty...")
    print("=" * 80)
    analyze_trajectory_uncertainty(
        dataset,
        model,
        n_trajectories=args.n_trajectories,
        n_posterior=args.n_posterior,
        output_path=output_dir / "uncertainty_trajectories.png",
    )

    print("\n" + "=" * 80)
    print("Uncertainty analysis complete!")
    print(f"Results saved to {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
