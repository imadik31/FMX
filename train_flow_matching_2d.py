"""
Training script for 2D toy datasets with Flow Matching variants.

Supports three loss types:
1. CFM (Conditional Flow Matching): Standard velocity regression at conditional bridge points.
   - Loss: MSE between v_θ(x_t, t) and (x₁ - x₀)
   - Fastest and most stable baseline

2. CFMX (Conditional Flux Matching): Flux matching at conditional bridge points.
   - Loss: Flux-MMD² between model and reference flux at x_t = (1-t)x₀ + tx₁
   - Like CFM but matches full flux (position + velocity) using operator-valued kernels
   - Recommended over FMX for better stability

3. FMX (Flux Matching): Non-conditional flux matching with pushed particles.
   - Loss: Flux-MMD² between pushed model particles and reference
   - Model particles obtained by integrating from x₀ to time t
   - More expensive (requires ODE integration) and less stable

Usage:
  python train_flow_matching_2d.py --dataset moons --loss cfmx --fmx_auto_sigma
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d
from torch import Tensor, nn
from torch.nn import Module

from flow_matching import visualization
from flow_matching.datasets import TOY_DATASETS
from flow_matching.solver import ModelWrapper
from flow_matching.utils import set_seed

from flow_matching.fmx import fmx_loss


class Swish(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(x)


class Mlp(Module):
    def __init__(self, dim: int = 2, time_dim: int = 1, h: int = 64) -> None:
        super().__init__()
        self.input_dim = dim
        self.time_dim = time_dim
        self.hidden_dim = h
        self.layers = nn.Sequential(
            nn.Linear(dim + time_dim, h),
            Swish(),
            nn.Linear(h, h),
            Swish(),
            nn.Linear(h, h),
            Swish(),
            nn.Linear(h, dim),
        )

    def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
        size = x_t.size()
        x_t = x_t.reshape(-1, self.input_dim)
        t = t.reshape(-1, self.time_dim).float()
        t = t.reshape(-1, 1).expand(x_t.size(0), 1)
        h = torch.cat([x_t, t], dim=1)
        output = self.layers(h)
        return output.reshape(*size)


def push_to_time(vnet: Mlp, x0: Tensor, t: Tensor, n_steps: int = 32) -> Tensor:
    """
    Simple fixed-step Euler integrator to push base samples x0 from time 0 to time t.
    x0: [B,d], t: [B,1] or [B]; returns x_t ~ p_{t,theta}.
    """
    if t.ndim == 1:
        t = t[:, None]
    # Use a scalar dt (same for the batch); you can switch to per-sample stepping if desired.
    dt = (t.max() - 0.0) / n_steps
    x = x0
    tau = 0.0
    for _ in range(n_steps):
        tau = tau + dt
        tau_tensor = torch.ones_like(x[:, :1]) * tau  # [B,1]
        x = x + dt * vnet(x_t=x, t=tau_tensor)
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=TOY_DATASETS.keys(), required=True)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--loss", choices=["cfm", "fmx", "cfmx"], default="cfm")
    parser.add_argument("--fmx_sigma", type=float, default=1.0)
    parser.add_argument("--fmx_auto_sigma", action="store_true", help="Auto-compute sigma using median heuristic")
    parser.add_argument("--fmx_steps", type=int, default=32)  # Euler steps to push model to time t (only for non-conditional FMX)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    # Save under .../<loss>/<dataset>/ to separate CFM vs FMX runs
    args.output_dir = Path(args.output_dir) / args.loss / args.dataset
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Dataset: {args.dataset}")

    # Training parameters
    learning_rate = 1e-3
    batch_size = 4096
    iterations = 20000
    log_every = 2000
    hidden_dim = 512

    dataset = TOY_DATASETS[args.dataset](device=device)

    flow = Mlp(dim=dataset.dim, time_dim=1, h=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(flow.parameters(), learning_rate)

    # Training
    losses = []
    for global_step in range(iterations):
        x_1 = dataset.sample(batch_size)                  # data samples
        x_0 = torch.randn_like(x_1).to(device)           # base samples ~ N(0,I)
        t = torch.rand(x_1.size(0), 1).to(device)        # times ~ U[0,1]

        # Conditional Flow Matching target (linear bridge as in the paper)
        # x_t^* = (1 - t) x0 + t x1, and v^* = x1 - x0
        x_t = (1 - t) * x_0 + t * x_1                    # reference positions at time t
        dx_t = x_1 - x_0                                 # reference velocities at those positions

        optimizer.zero_grad()

        if args.loss == "cfm":
            # Original CFM: regress velocity on reference points
            v_pred = flow(x_t=x_t, t=t)
            loss = F.mse_loss(v_pred, dx_t)

        elif args.loss == "cfmx":
            # -------- Conditional Flux Matching (CFMX) --------
            # Like CFM, we evaluate both reference and model at the conditional bridge points.
            # Reference flux from the linear bridge x_t = (1-t)x₀ + tx₁:
            Xr, vr = x_t.detach(), dx_t.detach()

            # Model flux at the SAME conditional bridge points (not pushed through the flow):
            Xm = x_t  # Evaluate at the same positions as reference
            vm = flow(x_t=Xm, t=t)  # Model velocity at conditional bridge points

            # Match fluxes using operator-valued kernel
            loss = fmx_loss(
                Xm=Xm.reshape(Xm.size(0), -1),
                vm=vm.reshape(vm.size(0), -1),
                Xr=Xr.reshape(Xr.size(0), -1),
                vr=vr.reshape(vr.size(0), -1),
                sigma=args.fmx_sigma,
                auto_sigma=args.fmx_auto_sigma,
            )

        else:  # args.loss == "fmx"
            # -------- Flux Matching (FMX) - non-conditional --------
            # Reference flux samples (positions & velocities) from the linear bridge:
            Xr, vr = x_t.detach(), dx_t.detach()

            # Model particles at same time t (push base x0 -> time t under current flow)
            with torch.no_grad():
                Xm = push_to_time(flow, x_0.detach(), t.detach(), n_steps=args.fmx_steps)

            vm = flow(x_t=Xm, t=t)  # model velocity at model particles

            # FMX operates on [B, d]; flatten (even though here d=2) for generality
            loss = fmx_loss(
                Xm=Xm.reshape(Xm.size(0), -1),
                vm=vm.reshape(vm.size(0), -1),
                Xr=Xr.reshape(Xr.size(0), -1),
                vr=vr.reshape(vr.size(0), -1),
                sigma=args.fmx_sigma,
                auto_sigma=args.fmx_auto_sigma,
            )

        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if (global_step + 1) % log_every == 0:
            print(f"| step: {global_step+1:6d} | loss: {loss.item():8.4f} |")

    flow.eval()
    torch.save(flow.state_dict(), Path(args.output_dir) / "ckpt.pth")

    # Plot learning curves

    steps = np.arange(1, len(losses) + 1)
    smoothed_losses = gaussian_filter1d(losses, sigma=5)
    blue = "#1f77b4"
    plt.figure(figsize=(6, 5))
    plt.plot(steps, losses, color=blue, alpha=0.3)
    plt.plot(steps, smoothed_losses, color=blue, linewidth=2)
    plt.title("Training dynamics", fontsize=16)
    plt.xlabel("Steps", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(Path(args.output_dir) / "losses.png")
    print("Training curves saved to", Path(args.output_dir) / "losses.png")

    # Sampling with ODE solver and visualization

    class WrappedModel(ModelWrapper):
        def forward(self, x: Tensor, t: Tensor, **extras) -> Tensor:
            return self.model(x_t=x, t=t)

    wrapped_model = WrappedModel(flow)

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


if __name__ == "__main__":
    main()
