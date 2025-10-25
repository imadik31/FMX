"""
Training script for 2D toy datasets with Flow Matching variants.

Supports three loss types:
1. CFM (Conditional Flow Matching): Standard velocity regression at conditional bridge points.
   - Loss: MSE between v_θ(x_t, t) and (x₁ - x₀)
   - Fastest and most stable baseline

2. FMX / CFMX (Flux Matching): Flux matching with pushed particles.
   - Reference flux: (Xr=x_t, vr=x₁-x₀) from conditional bridge
   - Model flux: (Xm, vm) where Xm is pushed from x₀ to time t via ODE
   - Objective: E[J,J] - 2*E[J,J*] (what we minimize, can be very negative)
   - Metric: E[J,J] - 2*E[J,J*] + E[J*,J*] (what we log, always >= 0)
   - Both FMX and CFMX use the same implementation

Key insight for FMX/CFMX:
  We minimize obj = E[J,J] - 2*E[J,J*] (dropping constant E[J*,J*] for efficiency).
  We log mmd2 = obj + E[J*,J*], which is the full non-negative MMD² metric.
  Both have the same gradients, so optimization is correct even though obj can be very negative.
  The logged mmd2 shows you the true non-negative distance being minimized.

Usage:
  # Recommended: CFMX with auto sigma
  python train_flow_matching_2d.py --dataset moons --loss cfmx --fmx_auto_sigma

  # Or with manual sigma
  python train_flow_matching_2d.py --dataset moons --loss cfmx --fmx_sigma 0.5

  # Baseline CFM for comparison
  python train_flow_matching_2d.py --dataset moons --loss cfm
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

from flow_matching.fmx import fmx_objective_and_metric


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
    Per-sample Euler integrator: push each x0[i] from time 0 to its own t[i].
    x0: [B,d], t: [B,1] or [B]; returns x_t ~ p_{t,theta}.
    """
    if t.ndim == 1:
        t = t[:, None]  # [B, 1]

    # Per-sample step size: each particle integrates to its own time
    dt = t / n_steps  # [B, 1]
    x = x0
    tau = torch.zeros_like(t)  # [B, 1] - start at time 0 for all

    for _ in range(n_steps):
        tau = tau + dt  # each sample advances toward its own t[i]
        x = x + dt * vnet(x_t=x, t=tau)

    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=TOY_DATASETS.keys(), required=True)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--loss", choices=["cfm", "fmx", "cfmx"], default="cfm")
    parser.add_argument("--fmx_sigma", type=float, default=1.0, help="Manual sigma (ignored if fmx_auto_sigma=True)")
    parser.add_argument("--fmx_auto_sigma", action="store_true", help="Auto-compute sigma using median heuristic")
    parser.add_argument("--fmx_steps", type=int, default=32, help="Euler steps for non-conditional FMX")
    parser.add_argument("--fmx_ridge", type=float, default=1e-6, help="Ridge regularization for kernel stability")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    # Save under .../<loss>/<dataset>/ to separate CFM vs FMX runs
    args.output_dir = Path(args.output_dir) / args.loss / args.dataset
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Dataset: {args.dataset}")

    # Training parameters
    # Lower LR for (C)FMX since Hessian kernels are sharper
    learning_rate = 5e-4 if args.loss in ("fmx", "cfmx") else 1e-3
    batch_size = 4096
    iterations = 20000
    log_every = 2000
    hidden_dim = 512

    dataset = TOY_DATASETS[args.dataset](device=device)

    flow = Mlp(dim=dataset.dim, time_dim=1, h=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(flow.parameters(), learning_rate)

    # EMA for sigma (stability for FMX/CFMX)
    sigma_ema = None
    ema_momentum = 0.9

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
            # -------- Conditional Flow Matching (CFM) --------
            # Standard velocity regression at conditional bridge points
            v_pred = flow(x_t=x_t, t=t)
            loss = F.mse_loss(v_pred, dx_t)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        elif args.loss in ("fmx", "cfmx"):
            # -------- Flux Matching (FMX / CFMX) --------
            # Reference flux from conditional bridge: positions Xr and velocities vr
            Xr, vr = x_t.detach(), dx_t.detach()

            # Model particles at time t (push through current flow)
            with torch.no_grad():
                Xm = push_to_time(flow, x_0.detach(), t.detach(), n_steps=args.fmx_steps)
            vm = flow(x_t=Xm, t=t)

            # Flatten to [B, d]
            Xm_f = Xm.reshape(Xm.size(0), -1)
            vm_f = vm.reshape(vm.size(0), -1)
            Xr_f = Xr.reshape(Xr.size(0), -1)
            vr_f = vr.reshape(vr.size(0), -1)

            # Compute sigma with EMA for stability
            if args.fmx_auto_sigma:
                from flow_matching.fmx import median_heuristic_sigma
                with torch.no_grad():
                    sigma_now = median_heuristic_sigma(Xm_f.detach(), Xr_f.detach())
                    sigma_now = max(0.5 * sigma_now, 1e-2)
                    if sigma_ema is None:
                        sigma_ema = sigma_now
                    else:
                        sigma_ema = ema_momentum * sigma_ema + (1 - ema_momentum) * sigma_now
                sigma_used = sigma_ema
            else:
                sigma_used = args.fmx_sigma

            # Compute objective and metric with fixed sigma
            obj, mmd2 = fmx_objective_and_metric(
                Xm_f, vm_f, Xr_f, vr_f,
                sigma=sigma_used,
                use_auto_sigma=False,  # sigma already computed above
                ridge=args.fmx_ridge,
            )

            # Optimize the objective
            loss = obj
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=5.0)
            optimizer.step()

            # Log the non-negative metric (not the objective!)
            mmd2_val = float(mmd2.detach().cpu())
            assert mmd2_val >= -1e-5, f"mmd2 should be non-negative, got {mmd2_val}"  # allow tiny numerical error
            losses.append(max(0.0, mmd2_val))  # clamp to 0 if slightly negative due to numerical error

        if (global_step + 1) % log_every == 0:
            # Print the last logged loss value
            print(f"| step: {global_step+1:6d} | loss: {losses[-1]:8.4f} |")

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
