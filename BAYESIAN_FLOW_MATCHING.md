# Bayesian Flow Matching

This document describes the Bayesian Flow Matching (BFM) implementation, which adds **uncertainty quantification** to Flow Matching models through last-layer variational inference.

## Overview

Bayesian Flow Matching extends Conditional Flow Matching (CFM) by:

1. **Replacing the last layer** with a variational Bayesian layer
2. **Adding KL divergence** regularization to the training objective (ELBO)
3. **Enabling uncertainty estimation** through posterior sampling during inference

This provides:
- **Epistemic uncertainty** quantification
- **Better calibration** on out-of-distribution data
- **Improved robustness** to data distribution shifts
- **Minimal computational overhead** during training

## Mathematical Framework

### Standard CFM

Standard Conditional Flow Matching learns a velocity field `v_θ(x_t, t)` by minimizing:

```
L_CFM = E_{x_0, x_1, t} [||v_θ(x_t, t) - (x_1 - x_0)||^2]
```

where `x_t = (1-t)x_0 + t·x_1` is the probability path.

### Bayesian Flow Matching

BFM decomposes the velocity network as:

```
v_θ(x, t) = W·φ_ψ(x, t) + b
```

where:
- `φ_ψ(x, t)` is a **deterministic backbone** (all layers except the last)
- `W, b` are the **last-layer weights** with Bayesian treatment

We place priors and posteriors:
- **Prior**: `p(W, b) = N(0, σ_p² I)`
- **Posterior**: `q(W, b) = N((W_μ, b_μ), diag(σ_W², σ_b²))`

The **ELBO objective** becomes:

```
L_BFM = 1/(2σ_ℓ²) · E_{q(W,b)} [||v_θ(x_t, t) - (x_1 - x_0)||^2] + β·KL(q||p)
```

where:
- `σ_ℓ` is the residual standard deviation (likelihood noise)
- `β` is the KL weight (regularization strength)
- KL term regularizes the posterior toward the prior

### Key Hyperparameters

| Parameter | Symbol | Recommended Range | Description |
|-----------|--------|-------------------|-------------|
| KL weight | `β` | `1e-5` to `1e-3` | Controls regularization strength |
| Residual std | `σ_ℓ` | `0.1` to `1.0` | Likelihood noise level |
| Prior std | `σ_p` | `0.5` to `2.0` | Prior belief about weight magnitude |
| Initial log-std | - | `-3.0` to `-1.0` | Initial posterior uncertainty |

## Implementation

### 1. Core Components

Located in `flow_matching/models/bayesian.py`:

#### `BayesianHead`
- Variational last layer with diagonal Gaussian posterior
- Implements `kl_divergence()` and `sample_forward()`
- Reparameterization trick for backpropagation

#### `BayesianMLP`
- Full Bayesian flow model for 2D datasets
- Deterministic backbone + Bayesian head
- Methods: `forward_sample()`, `forward_mean()`, `kl_divergence()`

### 2. Training Scripts

#### **Option 1: Full Bayesian Training** (`train_bayesian_flow_matching_2d.py`)

Train from scratch with variational inference:

```bash
python train_bayesian_flow_matching_2d.py \
    --dataset checkerboard \
    --beta 1e-4 \
    --sigma-likelihood 1.0 \
    --sigma-prior 1.0 \
    --beta-warmup-steps 2000 \
    --iterations 20000
```

**Key Arguments:**
- `--beta`: KL weight (higher = more regularization)
- `--sigma-likelihood`: Residual noise assumption
- `--sigma-prior`: Prior std for Bayesian head
- `--beta-warmup-steps`: Linearly warmup β from 0 to target value
- `--n-mc-train`: MC samples during training (default: 1)

**Outputs:**
- `outputs/bfm/{dataset}/ckpt.pth`: Trained model checkpoint
- `outputs/bfm/{dataset}/losses.png`: Training curves (total loss, MSE, KL, β schedule)
- `outputs/bfm/{dataset}/sampling_*.png`: Standard visualizations
- `outputs/bfm/{dataset}/vector_field_*.gif`: Animated flow field

#### **Option 2: Laplace Approximation** (`laplace_approximation.py`)

Convert a **pre-trained deterministic model** to Bayesian (no retraining):

```bash
# Step 1: Train standard CFM model
python train_flow_matching_2d.py --dataset checkerboard

# Step 2: Fit Laplace approximation
python laplace_approximation.py \
    --dataset checkerboard \
    --checkpoint outputs/cfm/checkerboard/ckpt.pth \
    --damping 1e-3 \
    --n-batches 50
```

**Key Arguments:**
- `--checkpoint`: Path to trained deterministic model
- `--damping`: Regularization for Fisher matrix inversion (numerical stability)
- `--n-batches`: Number of batches for computing empirical Fisher

**Outputs:**
- `outputs/laplace/{dataset}/laplace_params.pth`: Posterior parameters (means and variances)

**Pros:**
- Zero training changes (works with any trained CFM model)
- Very fast to compute
- Good calibration

**Cons:**
- Strictly local around the optimum
- Less expressive than full variational inference

### 3. Uncertainty Quantification (`sample_with_uncertainty.py`)

Visualize and analyze epistemic uncertainty:

```bash
python sample_with_uncertainty.py \
    --dataset checkerboard \
    --checkpoint outputs/bfm/checkerboard/ckpt.pth \
    --n-posterior 20 \
    --n-samples 1000 \
    --n-trajectories 5
```

**Key Arguments:**
- `--n-posterior`: Number of posterior samples (K) for uncertainty estimation
- `--n-samples`: Number of samples to generate
- `--n-trajectories`: Number of trajectories to analyze

**Outputs:**
- `uncertainty_samples.png`: Samples with uncertainty ellipses
- `uncertainty_trajectories.png`: Trajectory uncertainty evolution
- Console output with uncertainty statistics

**Visualizations:**
1. **True data distribution**
2. **Posterior mean samples**
3. **Samples with epistemic uncertainty ellipses** (2σ)
4. **Trajectory uncertainty over time**

## Example Workflow

### Full Pipeline (Recommended)

```bash
# 1. Train Bayesian Flow Matching
python train_bayesian_flow_matching_2d.py \
    --dataset checkerboard \
    --beta 1e-4 \
    --iterations 20000 \
    --hidden-dim 512

# 2. Analyze uncertainty
python sample_with_uncertainty.py \
    --dataset checkerboard \
    --checkpoint outputs/bfm/checkerboard/ckpt.pth \
    --n-posterior 20 \
    --n-samples 1000

# 3. Compare with standard CFM
python train_flow_matching_2d.py --dataset checkerboard
```

### Quick Start (Laplace Approximation)

```bash
# 1. Use existing CFM model
python laplace_approximation.py \
    --dataset checkerboard \
    --checkpoint outputs/cfm/checkerboard/ckpt.pth \
    --damping 1e-3

# 2. The Laplace posterior can be loaded and used for sampling
# (See laplace_approximation.py for LaplaceLastLayer API)
```

## Hyperparameter Tuning Guide

### KL Weight (β)

**Too small** (e.g., `1e-6`):
- Posterior collapses to deterministic solution
- Little/no uncertainty quantification
- Overfitting on small datasets

**Too large** (e.g., `1e-2`):
- Posterior too broad
- Poor fit to data
- Underfitting

**Recommended**: Start with `β = 1e-4` and adjust:
- Increase β if: model overfits, uncertainties too small
- Decrease β if: model underfits, training loss doesn't decrease

### Beta Warmup

**Why warmup?**
- Prevents early training collapse due to strong regularization
- Allows backbone to learn before Bayesian head kicks in

**Recommended**: Warmup over 10-20% of total iterations
- `--beta-warmup-steps 2000` for 20k iterations
- `--beta-warmup-steps 5000` for 50k iterations

### Prior Standard Deviation (σ_p)

Controls prior belief about weight magnitudes:
- **Smaller** (0.5): Stronger regularization, smaller weights
- **Larger** (2.0): Weaker regularization, larger weights

**Recommended**: Start with `σ_p = 1.0` and adjust based on:
- Data scale (normalize data to zero mean, unit variance)
- Network architecture (deeper networks may need smaller σ_p)

### Likelihood Standard Deviation (σ_ℓ)

Can be absorbed into β:
- Setting `σ_ℓ = 1.0` is common
- Alternatively, tune `β/(2σ_ℓ²)` as a single hyperparameter

## Benefits and Use Cases

### 1. Uncertainty Quantification
- **Epistemic uncertainty**: Uncertainty due to limited data
- **Calibration**: Uncertainty correlates with actual error
- **Out-of-distribution detection**: High uncertainty in OOD regions

### 2. Robustness
- **Data distribution shifts**: Better generalization on shifted data
- **Small data regime**: Regularization prevents overfitting
- **Noisy data**: Uncertainty captures noise level

### 3. Active Learning
- **Sample selection**: Query points with high uncertainty
- **Data efficiency**: Focus labeling effort on uncertain regions

### 4. Model Selection
- **Hyperparameter tuning**: Monitor KL divergence for convergence
- **Architecture search**: Compare uncertainty quality across models

## Performance Considerations

### Training
- **Overhead**: ~10-20% slower than standard CFM (due to KL computation)
- **Memory**: Slightly higher (stores posterior variances)
- **Convergence**: May need more iterations with high β

### Inference
- **Single sample**: Same speed as deterministic model
- **K samples**: K× slower (linear in number of posterior samples)
- **Recommendation**: Use `K=3-5` for images, `K=10-20` for low-dim data

## Theoretical Guarantees

### ELBO Maximization
The BFM objective is a valid ELBO (Evidence Lower Bound):

```
log p(x_1|x_0, t) ≥ -L_BFM
```

Maximizing the ELBO corresponds to minimizing KL divergence from the true posterior.

### Asymptotic Behavior
As `N → ∞` (infinite data):
- Posterior concentrates around MAP (maximum a posteriori)
- Bayesian and deterministic solutions coincide
- Uncertainty decreases as `O(1/√N)`

## Comparison with Standard CFM

| Aspect | Standard CFM | Bayesian FM | Laplace FM |
|--------|-------------|-------------|------------|
| **Training** | Fast | +10-20% overhead | Same as CFM |
| **Uncertainty** | None | Full epistemic | Full epistemic |
| **Retraining** | Required | Required | Not required |
| **Expressivity** | Deterministic | Stochastic | Local stochastic |
| **Calibration** | N/A | Excellent | Excellent |
| **Use case** | Standard | High-stakes, OOD | Retrofit existing |

## Advanced Topics

### Converting Deterministic Models

Use `convert_to_bayesian()` to retrofit existing models:

```python
from flow_matching.models.bayesian import convert_to_bayesian

# Load trained deterministic model
model = Mlp(dim=2, h=512)
model.load_state_dict(torch.load("ckpt.pth"))

# Convert to Bayesian
bayesian_model = convert_to_bayesian(
    model,
    sigma_p=1.0,
    init_log_sigma=-2.0
)

# Fine-tune with BFM objective
# ... (see train_bayesian_flow_matching_2d.py)
```

### Multi-MC Training

For very small datasets, use multiple MC samples during training:

```bash
python train_bayesian_flow_matching_2d.py \
    --dataset checkerboard \
    --n-mc-train 5 \
    --batch-size 1024  # Reduce batch size to fit in memory
```

**Trade-off**: Better gradient estimates but slower training.

### Adaptive Beta Scheduling

Dynamically adjust β based on KL divergence:

```python
# Target KL divergence
target_kl = 100.0

# Adjust beta to match target
if kl < target_kl:
    beta *= 1.1  # Increase regularization
elif kl > target_kl * 1.5:
    beta *= 0.9  # Decrease regularization
```

## Troubleshooting

### Issue: Uncertainties collapse to zero

**Causes:**
- β too small
- Too many training iterations
- Over-regularized backbone

**Solutions:**
- Increase β (try `1e-3` or `1e-2`)
- Reduce training iterations
- Add warmup for β
- Use higher `init_log_sigma` (e.g., `-1.0`)

### Issue: Model doesn't fit data well

**Causes:**
- β too large
- Poor backbone architecture
- Warmup too short

**Solutions:**
- Decrease β (try `1e-5` or `1e-6`)
- Increase hidden dimensions
- Extend warmup period
- Train deterministic CFM first to verify architecture

### Issue: NaN losses during training

**Causes:**
- Numerical instability in KL computation
- Exploding log-std parameters

**Solutions:**
- Add gradient clipping (already in script: `max_norm=1.0`)
- Lower learning rate
- Initialize with smaller `init_log_sigma` (e.g., `-3.0`)
- Check for `inf`/`nan` in data

## References

### Bayesian Deep Learning
- [Weight Uncertainty in Neural Networks (Bayes by Backprop)](https://arxiv.org/abs/1505.05424)
- [Practical Deep Learning with Bayesian Principles](https://arxiv.org/abs/1906.02506)

### Flow Matching
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [Improving and Generalizing Flow-Based Generative Models](https://arxiv.org/abs/2302.00482)

### Laplace Approximation
- [Laplace Redux — Effortless Bayesian Deep Learning](https://arxiv.org/abs/2106.14806)
- [A Simple Baseline for Bayesian Uncertainty in Deep Learning](https://arxiv.org/abs/1902.02476)

## Citation

If you use Bayesian Flow Matching in your research, please cite:

```bibtex
@article{lipman2023flow,
  title={Flow matching for generative modeling},
  author={Lipman, Yaron and Chen, Ricky TQ and Ben-Hamu, Heli and Nickel, Maximilian and Le, Matthew},
  journal={arXiv preprint arXiv:2210.02747},
  year={2023}
}

@article{blundell2015weight,
  title={Weight uncertainty in neural networks},
  author={Blundell, Charles and Cornebise, Julien and Kavukcuoglu, Koray and Wierstra, Daan},
  journal={International Conference on Machine Learning},
  year={2015}
}
```

## License

This implementation follows the same license as the parent FMX repository.
