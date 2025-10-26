"""
Bayesian Flow Matching training on 2D synthetic datasets.

This script implements Bayesian Flow Matching with last-layer variational inference.
It adds uncertainty quantification to the standard CFM objective through:
1. A Bayesian last layer with diagonal Gaussian posterior
2. KL divergence regularization from the ELBO
3. Uncertainty-aware sampling and visualization
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d
from torch import Tensor

from flow_matching import visualization
from flow_matching.datasets import TOY_DATASETS
from flow_matching.models.bayesian import BayesianMLP
from flow_matching.solver import ModelWrapper
from flow_matching.utils import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=TOY_DATASETS.keys(), required=True)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=3)

    # Bayesian-specific hyperparameters
    parser.add_argument("--beta", type=float, default=1e-4,
                        help="KL weight (1e-5 to 1e-3 recommended)")
    parser.add_argument("--sigma-likelihood", type=float, default=1.0,
                        help="Residual std for likelihood")
    parser.add_argument("--sigma-prior", type=float, default=1.0,
                        help="Prior std for Bayesian head")
    parser.add_argument("--n-mc-train", type=int, default=1,
                        help="Number of MC samples during training")
    parser.add_argument("--beta-warmup-steps", type=int, default=2000,
                        help="Number of steps to warmup KL weight from 0 to beta")

    # Sampling hyperparameters
    parser.add_argument("--n-mc-sample", type=int, default=10,
                        help="Number of posterior samples for uncertainty quantification")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    args.output_dir = Path(args.output_dir) / "bfm" / args.dataset
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Bayesian Flow Matching Training")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Hidden dim: {args.hidden_dim}, Layers: {args.num_layers}")
    print(f"Batch size: {args.batch_size}, Iterations: {args.iterations}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Beta (KL weight): {args.beta}, Warmup steps: {args.beta_warmup_steps}")
    print(f"Sigma likelihood: {args.sigma_likelihood}, Sigma prior: {args.sigma_prior}")
    print(f"MC samples (train): {args.n_mc_train}, MC samples (eval): {args.n_mc_sample}")
    print("=" * 80)

    dataset = TOY_DATASETS[args.dataset](device=device)

    # Create Bayesian flow model
    flow = BayesianMLP(
        dim=dataset.dim,
        time_dim=1,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        sigma_p=args.sigma_prior,
        init_log_sigma=-2.0,
    ).to(device)

    optimizer = torch.optim.AdamW(flow.parameters(), args.learning_rate)

    # Training metrics
    losses = []
    mse_losses = []
    kl_losses = []
    beta_schedule = []

    print("\nTraining...")
    for global_step in range(args.iterations):
        # Sample data
        x_1 = dataset.sample(args.batch_size)
        x_0 = torch.randn_like(x_1).to(device)
        t = torch.rand(x_1.size(0), 1).to(device)

        # Compute probability path and target velocity
        # x_t = (1-t) * x_0 + t * x_1
        # v* = x_1 - x_0
        x_t = (1 - t) * x_0 + t * x_1
        dx_t = x_1 - x_0

        # Compute KL weight with warmup
        if args.beta_warmup_steps > 0:
            beta_t = args.beta * min(1.0, global_step / args.beta_warmup_steps)
        else:
            beta_t = args.beta
        beta_schedule.append(beta_t)

        optimizer.zero_grad()

        # Forward pass with MC sampling
        v_pred_mc = flow.forward_sample(x_t, t, n_mc=args.n_mc_train)  # [n_mc, B, d]

        # Compute ELBO loss
        # L = 1/(2*sigma_l^2) * E_q[||v - v*||^2] + beta * KL(q||p)
        mse = ((v_pred_mc - dx_t).pow(2)).mean()
        kl = flow.kl_divergence()

        likelihood_term = (1.0 / (2 * args.sigma_likelihood ** 2)) * mse
        loss = likelihood_term + beta_t * kl

        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
        optimizer.step()

        # Log metrics
        losses.append(loss.item())
        mse_losses.append(mse.item())
        kl_losses.append(kl.item())

        if (global_step + 1) % 2000 == 0:
            print(
                f"| step: {global_step+1:6d} | "
                f"loss: {loss.item():8.4f} | "
                f"mse: {mse.item():8.4f} | "
                f"kl: {kl.item():8.2f} | "
                f"beta: {beta_t:.6f} |"
            )

    flow.eval()

    # Save model with configuration
    checkpoint = {
        'state_dict': flow.state_dict(),
        'config': {
            'dim': dataset.dim,
            'time_dim': 1,
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers,
            'sigma_p': args.sigma_prior,
        }
    }
    torch.save(checkpoint, Path(args.output_dir) / "ckpt.pth")
    print(f"\nModel saved to {Path(args.output_dir) / 'ckpt.pth'}")

    # Plot learning curves
    print("\nPlotting learning curves...")
    steps = np.arange(1, len(losses) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Total loss
    ax = axes[0, 0]
    smoothed = gaussian_filter1d(losses, sigma=5)
    ax.plot(steps, losses, alpha=0.3, color="#1f77b4")
    ax.plot(steps, smoothed, linewidth=2, color="#1f77b4")
    ax.set_title("Total Loss (ELBO)", fontsize=14)
    ax.set_xlabel("Steps", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.grid(True, alpha=0.3)

    # MSE (likelihood term)
    ax = axes[0, 1]
    smoothed = gaussian_filter1d(mse_losses, sigma=5)
    ax.plot(steps, mse_losses, alpha=0.3, color="#ff7f0e")
    ax.plot(steps, smoothed, linewidth=2, color="#ff7f0e")
    ax.set_title("MSE Loss (Likelihood)", fontsize=14)
    ax.set_xlabel("Steps", fontsize=12)
    ax.set_ylabel("MSE", fontsize=12)
    ax.grid(True, alpha=0.3)

    # KL divergence
    ax = axes[1, 0]
    smoothed = gaussian_filter1d(kl_losses, sigma=5)
    ax.plot(steps, kl_losses, alpha=0.3, color="#2ca02c")
    ax.plot(steps, smoothed, linewidth=2, color="#2ca02c")
    ax.set_title("KL Divergence", fontsize=14)
    ax.set_xlabel("Steps", fontsize=12)
    ax.set_ylabel("KL(q||p)", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Beta schedule
    ax = axes[1, 1]
    ax.plot(steps, beta_schedule, linewidth=2, color="#d62728")
    ax.set_title("Beta Schedule (KL weight)", fontsize=14)
    ax.set_xlabel("Steps", fontsize=12)
    ax.set_ylabel("Beta", fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(args.output_dir) / "losses.png", dpi=150)
    print(f"Training curves saved to {Path(args.output_dir) / 'losses.png'}")

    # Sampling with ODE solver and visualization
    print("\nGenerating visualizations...")

    # Wrapper for posterior mean (deterministic sampling)
    class BayesianModelWrapper(ModelWrapper):
        def __init__(self, model, use_mean=False):
            super().__init__(model)
            self.use_mean = use_mean

        def forward(self, x: Tensor, t: Tensor, **extras) -> Tensor:
            if self.use_mean:
                return self.model.forward_mean(x_t=x, t=t)
            else:
                # Single MC sample
                return self.model.forward_sample(x_t=x, t=t, n_mc=1).squeeze(0)

    # Use posterior mean for deterministic visualization
    wrapped_model = BayesianModelWrapper(flow, use_mean=True)

    visualization.plot_ode_sampling_evolution(
        flow=wrapped_model,
        dataset=dataset,
        output_dir=args.output_dir,
        filename=f"sampling_{args.dataset}_w_solver.png",
    )

    visualization.save_vector_field_and_samples_as_gif(
        flow=wrapped_model,
        dataset=dataset,
        output_dir=args.output_dir,
        filename=f"vector_field_{args.dataset}.gif",
    )

    visualization.plot_likelihood(
        flow=wrapped_model,
        dataset=dataset,
        output_dir=args.output_dir,
        filename=f"likelihood_{args.dataset}.png",
    )

    print("\nTraining complete!")
    print(f"All outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
