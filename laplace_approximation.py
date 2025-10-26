"""
Laplace Last-Layer Approximation (LLLA) for Flow Matching.

This script provides a simpler alternative to full Bayesian training:
1. Take a trained deterministic CFM model
2. Fit a Gaussian posterior over the last layer using Laplace approximation
3. Sample from this posterior for uncertainty quantification

Advantages:
- Zero training changes (works with any trained model)
- Very fast to compute
- Good calibration
- Easy to apply to existing models

Disadvantages:
- Strictly local around the optimum
- Less expressive than variational inference
"""

import argparse
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from flow_matching.datasets import TOY_DATASETS
from flow_matching.utils import set_seed


class LaplaceLastLayer:
    """
    Laplace approximation for the last linear layer.

    Fits a Gaussian posterior q(W, b) = N(mu, Sigma) where:
    - mu = trained weights (MAP estimate)
    - Sigma^-1 = Hessian + lambda * I (with damping for numerical stability)

    For efficiency, we use a diagonal approximation (empirical Fisher).
    """

    def __init__(
        self,
        model: nn.Module,
        damping: float = 1e-3,
        device: torch.device = None,
    ):
        """
        Args:
            model: Trained deterministic model with at least one Linear layer
            damping: Regularization for Hessian inversion (lambda)
            device: Device for computation
        """
        self.device = device or next(model.parameters()).device
        self.model = model.to(self.device)
        self.damping = damping

        # Find last linear layer
        self.last_linear = None
        self.last_linear_name = None

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                self.last_linear = module
                self.last_linear_name = name

        if self.last_linear is None:
            raise ValueError("Model must have at least one Linear layer")

        print(f"Found last linear layer: {self.last_linear_name}")
        print(f"  Input features: {self.last_linear.in_features}")
        print(f"  Output features: {self.last_linear.out_features}")

        # Store mean (MAP estimate)
        self.W_mu = self.last_linear.weight.data.clone()  # [out, in]
        self.b_mu = self.last_linear.bias.data.clone()  # [out]

        # Posterior variance (to be computed)
        self.W_var = None  # [out, in]
        self.b_var = None  # [out]

    def fit(
        self,
        dataset,
        batch_size: int = 2048,
        n_batches: int = 50,
    ):
        """
        Fit Laplace approximation using empirical Fisher (diagonal).

        The empirical Fisher is computed as:
        F_ij ≈ E[(∂log p(y|x,θ)/∂θ_i)(∂log p(y|x,θ)/∂θ_j)]

        For regression with Gaussian likelihood:
        F ≈ E[(∂L/∂θ)^2] where L = MSE loss

        Args:
            dataset: Dataset for computing empirical Fisher
            batch_size: Batch size for Fisher computation
            n_batches: Number of batches to average over
        """
        print(f"\nFitting Laplace approximation...")
        print(f"  Batch size: {batch_size}, Batches: {n_batches}")
        print(f"  Damping: {self.damping}")

        self.model.eval()

        # Accumulators for squared gradients
        W_grad_sq = torch.zeros_like(self.W_mu)
        b_grad_sq = torch.zeros_like(self.b_mu)

        n_samples = 0

        for batch_idx in tqdm(range(n_batches), desc="Computing Fisher"):
            # Sample data
            x_1 = dataset.sample(batch_size)
            x_0 = torch.randn_like(x_1).to(self.device)
            t = torch.rand(x_1.size(0), 1).to(self.device)

            # CFM objective
            x_t = (1 - t) * x_0 + t * x_1
            dx_t = x_1 - x_0

            # Forward pass
            self.model.zero_grad()

            # Get prediction
            if hasattr(self.model, "forward"):
                # Handle different forward signatures
                try:
                    pred = self.model(x_t=x_t, t=t)
                except TypeError:
                    pred = self.model(x_t, t)
            else:
                pred = self.model(x_t, t)

            # MSE loss
            loss = F.mse_loss(pred, dx_t)

            # Backward to get gradients
            loss.backward()

            # Accumulate squared gradients for last layer only
            if self.last_linear.weight.grad is not None:
                W_grad_sq += self.last_linear.weight.grad.data ** 2
            if self.last_linear.bias.grad is not None:
                b_grad_sq += self.last_linear.bias.grad.data ** 2

            n_samples += x_1.size(0)

        # Average over batches
        W_grad_sq /= n_batches
        b_grad_sq /= n_batches

        # Variance = 1 / (Fisher + damping)
        # For diagonal Fisher: var_i = 1 / (F_ii + lambda)
        self.W_var = 1.0 / (W_grad_sq + self.damping)
        self.b_var = 1.0 / (b_grad_sq + self.damping)

        print(f"Laplace approximation fitted!")
        print(f"  Mean W variance: {self.W_var.mean().item():.6f}")
        print(f"  Mean b variance: {self.b_var.mean().item():.6f}")
        print(f"  Max W variance: {self.W_var.max().item():.6f}")
        print(f"  Max b variance: {self.b_var.max().item():.6f}")

    def sample_weights(self) -> Tuple[Tensor, Tensor]:
        """
        Sample from posterior q(W, b).

        Returns:
            W_sample: [out, in]
            b_sample: [out]
        """
        if self.W_var is None or self.b_var is None:
            raise RuntimeError("Must call fit() before sampling")

        # Sample from N(mu, var)
        W_sample = self.W_mu + torch.randn_like(self.W_mu) * torch.sqrt(self.W_var)
        b_sample = self.b_mu + torch.randn_like(self.b_mu) * torch.sqrt(self.b_var)

        return W_sample, b_sample

    def forward_sample(self, *args, n_mc: int = 1, **kwargs) -> Tensor:
        """
        Forward pass with sampled weights.

        Returns:
            Predictions [n_mc, batch_size, out_features]
        """
        if self.W_var is None or self.b_var is None:
            raise RuntimeError("Must call fit() before sampling")

        samples = []

        for _ in range(n_mc):
            # Sample weights
            W_sample, b_sample = self.sample_weights()

            # Temporarily replace weights
            original_W = self.last_linear.weight.data.clone()
            original_b = self.last_linear.bias.data.clone()

            self.last_linear.weight.data = W_sample
            self.last_linear.bias.data = b_sample

            # Forward pass
            with torch.no_grad():
                if hasattr(self.model, "forward"):
                    try:
                        pred = self.model(*args, **kwargs)
                    except TypeError:
                        # Try alternative signature
                        pred = self.model(x_t=args[0], t=args[1])
                else:
                    pred = self.model(*args, **kwargs)

            samples.append(pred)

            # Restore original weights
            self.last_linear.weight.data = original_W
            self.last_linear.bias.data = original_b

        return torch.stack(samples, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=TOY_DATASETS.keys(), required=True)
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained deterministic CFM model",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/laplace")
    parser.add_argument("--damping", type=float, default=1e-3, help="Regularization for Fisher")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--n-batches", type=int, default=50, help="Batches for Fisher")
    parser.add_argument("--n-mc-samples", type=int, default=10, help="MC samples for testing")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    output_dir = Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Laplace Last-Layer Approximation for Flow Matching")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Damping: {args.damping}")
    print("=" * 80)

    # Load dataset
    dataset = TOY_DATASETS[args.dataset](device=device)

    # Load trained deterministic model
    # Need to reconstruct the model architecture
    from train_flow_matching_2d import Mlp

    model = Mlp(dim=dataset.dim, time_dim=1, h=512).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    print(f"Loaded deterministic model from {args.checkpoint}")

    # Fit Laplace approximation
    laplace = LaplaceLastLayer(model, damping=args.damping, device=device)
    laplace.fit(dataset, batch_size=args.batch_size, n_batches=args.n_batches)

    # Save Laplace parameters
    laplace_state = {
        "W_mu": laplace.W_mu,
        "b_mu": laplace.b_mu,
        "W_var": laplace.W_var,
        "b_var": laplace.b_var,
        "damping": laplace.damping,
    }
    torch.save(laplace_state, output_dir / "laplace_params.pth")
    print(f"\nLaplace parameters saved to {output_dir / 'laplace_params.pth'}")

    # Test sampling with uncertainty
    print("\nTesting uncertainty sampling...")
    x_1 = dataset.sample(100)
    x_0 = torch.randn_like(x_1).to(device)
    t = torch.rand(x_1.size(0), 1).to(device)
    x_t = (1 - t) * x_0 + t * x_1

    # Sample from posterior
    samples = laplace.forward_sample(x_t, t, n_mc=args.n_mc_samples)  # [K, B, d]

    mean = samples.mean(dim=0)
    std = samples.std(dim=0)

    print(f"  Mean prediction shape: {mean.shape}")
    print(f"  Std prediction shape: {std.shape}")
    print(f"  Mean uncertainty (L2): {std.norm(dim=-1).mean().item():.6f}")
    print(f"  Max uncertainty (L2): {std.norm(dim=-1).max().item():.6f}")

    print("\n" + "=" * 80)
    print("Laplace approximation complete!")
    print(f"Results saved to {output_dir}")
    print("=" * 80)
    print("\nTo use this for uncertainty quantification, load the laplace_params.pth")
    print("and sample from the posterior using LaplaceLastLayer.forward_sample()")


if __name__ == "__main__":
    main()
