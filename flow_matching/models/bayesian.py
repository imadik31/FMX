"""
Bayesian Flow Matching components with variational inference.

This module implements:
1. BayesianHead: Last-layer variational inference with diagonal Gaussian posterior
2. BayesianMLP: Bayesian flow matching model for 2D datasets
3. Utilities for KL divergence computation and uncertainty sampling
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BayesianHead(nn.Module):
    """
    Variational last layer with diagonal Gaussian posterior.

    Args:
        in_features: Input dimension (hidden size)
        out_features: Output dimension (data dimension)
        sigma_p: Prior standard deviation (default: 1.0)
        init_log_sigma: Initial log std for posterior (default: -2.0, i.e., std ≈ 0.135)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        sigma_p: float = 1.0,
        init_log_sigma: float = -2.0,
    ) -> None:
        super().__init__()

        # Mean parameters (initialize like standard linear layer)
        self.W_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.b_mu = nn.Parameter(torch.zeros(out_features))
        nn.init.kaiming_uniform_(self.W_mu, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W_mu)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.b_mu, -bound, bound)

        # Log-std parameters (diagonal posterior)
        self.W_logsig = nn.Parameter(torch.full((out_features, in_features), init_log_sigma))
        self.b_logsig = nn.Parameter(torch.full((out_features,), init_log_sigma))

        # Prior variance
        self.sigma_p2 = float(sigma_p ** 2)
        self.register_buffer("_sigma_p", torch.tensor(sigma_p))

    def kl_divergence(self) -> Tensor:
        """
        Compute KL(q(W,b) || p(W,b)) where:
        - q is the diagonal Gaussian posterior
        - p is N(0, sigma_p^2 I) prior

        Returns:
            Scalar KL divergence
        """
        # Variance from log-std
        W_var = torch.exp(2 * self.W_logsig)
        b_var = torch.exp(2 * self.b_logsig)

        # KL for Gaussian: 0.5 * [trace(Sigma_p^-1 Sigma_q) + mu^T Sigma_p^-1 mu - k + log(det(Sigma_p)/det(Sigma_q))]
        # For diagonal: 0.5 * sum[(sigma_q^2 + mu^2) / sigma_p^2 - 1 + 2*log(sigma_p/sigma_q)]

        kl_W = 0.5 * torch.sum(
            (W_var + self.W_mu ** 2) / self.sigma_p2
            - 1.0
            - 2 * self.W_logsig
            + 2 * math.log(self._sigma_p.item())
        )

        kl_b = 0.5 * torch.sum(
            (b_var + self.b_mu ** 2) / self.sigma_p2
            - 1.0
            - 2 * self.b_logsig
            + 2 * math.log(self._sigma_p.item())
        )

        return kl_W + kl_b

    def sample_forward(self, feats: Tensor) -> Tensor:
        """
        Sample from posterior and compute linear transformation.

        Args:
            feats: Input features [batch_size, in_features]

        Returns:
            Output [batch_size, out_features]
        """
        # Sample from q(W, b) using reparameterization trick
        eps_W = torch.randn_like(self.W_mu)
        eps_b = torch.randn_like(self.b_mu)

        W = self.W_mu + torch.exp(self.W_logsig) * eps_W
        b = self.b_mu + torch.exp(self.b_logsig) * eps_b

        return F.linear(feats, W, b)

    def forward_mean(self, feats: Tensor) -> Tensor:
        """
        Forward pass using posterior mean (no sampling).

        Args:
            feats: Input features [batch_size, in_features]

        Returns:
            Output [batch_size, out_features]
        """
        return F.linear(feats, self.W_mu, self.b_mu)


class BayesianMLP(nn.Module):
    """
    Bayesian Flow Matching MLP with last-layer variational inference.

    Architecture:
    - Deterministic backbone (multiple hidden layers)
    - Bayesian last layer (variational)

    Args:
        dim: Data dimension
        time_dim: Time dimension (default: 1)
        hidden_dim: Hidden layer size (default: 512)
        num_layers: Number of hidden layers (default: 3)
        sigma_p: Prior std for Bayesian head (default: 1.0)
        init_log_sigma: Initial log std for posterior (default: -2.0)
    """

    def __init__(
        self,
        dim: int = 2,
        time_dim: int = 1,
        hidden_dim: int = 512,
        num_layers: int = 3,
        sigma_p: float = 1.0,
        init_log_sigma: float = -2.0,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim

        # Build deterministic backbone
        layers = []
        in_dim = dim + time_dim

        for i in range(num_layers):
            layers.extend([
                nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim),
                nn.SiLU(),  # Swish activation
            ])

        self.backbone = nn.Sequential(*layers)

        # Bayesian last layer
        self.head = BayesianHead(
            in_features=hidden_dim,
            out_features=dim,
            sigma_p=sigma_p,
            init_log_sigma=init_log_sigma,
        )

    def encode(self, x_t: Tensor, t: Tensor) -> Tensor:
        """
        Encode (x_t, t) through deterministic backbone.

        Args:
            x_t: State at time t [batch_size, dim]
            t: Time [batch_size, time_dim] or [batch_size] or scalar

        Returns:
            Hidden features [batch_size, hidden_dim]
        """
        # Handle different time tensor shapes
        if t.dim() == 0:
            # Scalar time: expand to match batch size
            t = t.unsqueeze(0).unsqueeze(0).expand(x_t.size(0), 1)
        elif t.dim() == 1:
            # 1D time: add feature dimension
            t = t.unsqueeze(-1)

        h = torch.cat([x_t, t], dim=-1)
        return self.backbone(h)

    def forward_sample(self, x_t: Tensor, t: Tensor, n_mc: int = 1) -> Tensor:
        """
        Forward pass with Monte Carlo sampling from posterior.

        Args:
            x_t: State at time t [batch_size, dim]
            t: Time [batch_size, time_dim] or [batch_size]
            n_mc: Number of Monte Carlo samples (default: 1)

        Returns:
            Velocity predictions [n_mc, batch_size, dim]
        """
        feats = self.encode(x_t, t)

        # Sample n_mc times from posterior
        samples = torch.stack([self.head.sample_forward(feats) for _ in range(n_mc)], dim=0)
        return samples

    def forward_mean(self, x_t: Tensor, t: Tensor) -> Tensor:
        """
        Forward pass using posterior mean (deterministic).

        Args:
            x_t: State at time t [batch_size, dim]
            t: Time [batch_size, time_dim] or [batch_size]

        Returns:
            Velocity prediction [batch_size, dim]
        """
        feats = self.encode(x_t, t)
        return self.head.forward_mean(feats)

    def kl_divergence(self) -> Tensor:
        """Compute total KL divergence of Bayesian layers."""
        return self.head.kl_divergence()

    # Compatibility with existing interface
    def forward(self, x_t: Tensor, t: Tensor, **kwargs) -> Tensor:
        """
        Default forward (uses single MC sample).
        For training, use forward_sample() explicitly.
        """
        return self.forward_sample(x_t, t, n_mc=1).squeeze(0)


class DeterministicBackbone(nn.Module):
    """
    Extract deterministic backbone from a trained standard model.
    Used for Laplace approximation.

    Args:
        model: Trained deterministic model with Linear final layer
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()

        # Find all modules except the last Linear layer
        modules = []
        last_linear = None

        for name, module in model.named_children():
            if isinstance(module, nn.Linear):
                last_linear = module
            else:
                modules.append(module)

        if last_linear is None:
            raise ValueError("Model must have at least one Linear layer")

        self.backbone = nn.Sequential(*modules[:-1]) if len(modules) > 1 else modules[0]
        self.out_features = last_linear.out_features
        self.in_features = last_linear.in_features

    def forward(self, *args, **kwargs) -> Tensor:
        return self.backbone(*args, **kwargs)


def convert_to_bayesian(
    model: nn.Module,
    sigma_p: float = 1.0,
    init_log_sigma: float = -2.0,
) -> nn.Module:
    """
    Convert a trained deterministic model to Bayesian by replacing last layer.

    Args:
        model: Trained deterministic model
        sigma_p: Prior std for Bayesian head
        init_log_sigma: Initial log std for posterior

    Returns:
        Model with Bayesian last layer
    """
    # Find last linear layer
    last_linear = None
    last_linear_name = None

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last_linear = module
            last_linear_name = name

    if last_linear is None:
        raise ValueError("Model must have at least one Linear layer")

    # Create Bayesian head
    bayesian_head = BayesianHead(
        in_features=last_linear.in_features,
        out_features=last_linear.out_features,
        sigma_p=sigma_p,
        init_log_sigma=init_log_sigma,
    )

    # Initialize mean with trained weights
    bayesian_head.W_mu.data.copy_(last_linear.weight.data)
    bayesian_head.b_mu.data.copy_(last_linear.bias.data)

    # Replace last layer
    parent_module = model
    if "." in last_linear_name:
        parent_name = ".".join(last_linear_name.split(".")[:-1])
        parent_module = dict(model.named_modules())[parent_name]
        attr_name = last_linear_name.split(".")[-1]
    else:
        attr_name = last_linear_name

    setattr(parent_module, attr_name, bayesian_head)

    return model
