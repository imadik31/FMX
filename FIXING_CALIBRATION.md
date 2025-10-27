# Fixing Under-Calibrated Uncertainty in BFM

## Problem Diagnosis

Your results show severe under-calibration:
- **Spearman ρ = -0.50**: Negative correlation (uncertainty anti-correlated with error!)
- **Coverage ~3-10%** instead of 50-95%: Posterior is way too confident
- **KL decreased** from 1600 → 710: Posterior collapsed during training

## Root Causes & Fixes

1. **KL not normalized by parameter count** (FIXED ✓)
   - Previous: `kl = sum over all parameters` (~1600 for your model)
   - Now: `kl = (sum over all parameters) / num_params` (~1.5)
   - **Ratio**: 1600 / 1024 ≈ 1.56 per parameter
   - **Impact**: With β=1e-4, old effective regularization was ~0.16!

2. **Coverage test was flawed** (FIXED ✓)
   - Previous: per-dimension marginal intervals
   - Now: joint multivariate ellipsoid test
   - **Impact**: Coverage numbers will be HIGHER (more accurate)

3. **β may need re-tuning**
   - Old β=1e-4 was actually acting like β≈0.16 due to no normalization
   - New β=1e-4 × 1.5 ≈ 0.00015 effective regularization
   - **Recommendation**: Try β=0.05-0.2 with normalized KL

## Quick Fixes

### Fix 1: Retrain with Normalized KL (REQUIRED)

Your old model was trained without KL normalization. Pull the fixes and retrain:

```bash
git pull --ff-only  # Get the normalization fix

python train_bayesian_flow_matching_2d.py \
    --dataset checkerboard \
    --beta 0.1 \               # Start with 0.1 (was effectively 0.16 before)
    --iterations 20000 \
    --beta-warmup-steps 5000
```

**Expected behavior:**
- KL/p (per-parameter) should be ~1-2, not ~1600
- Positive ρ (0.2-0.4)
- Joint coverage within ±10% of expected
- W_std stays > 0.15

### Fix 2: Lower β if Still Over-Regularized

If Fix 1 still shows collapsing (W_std < 0.10):

```bash
python train_bayesian_flow_matching_2d.py \
    --dataset checkerboard \
    --beta 0.01 \              # Lower
    --iterations 20000 \
    --beta-warmup-steps 5000
```

### Fix 3: No Warmup (Debug)

To check if warmup is the issue:

```bash
python train_bayesian_flow_matching_2d.py \
    --dataset checkerboard \
    --beta 1e-5 \
    --iterations 20000 \
    --beta-warmup-steps 0      # No warmup
```

## Monitoring Training

With the updated training script (KL normalized), you'll see:

```
Model architecture:
  Bayesian head parameters: 1026
  (KL will be normalized by this count)

| step:   2000 | loss:   2.28 | mse:   4.25 | kl/p:  1.5623 | beta: 0.100000 | W_std: 0.1500 | b_std: 0.0100 |
```

**Key differences from old version:**
- **kl/p** instead of **kl**: Now per-parameter (~1.5 instead of ~1600)
- **β range**: Now 0.01-0.5 instead of 1e-5 to 1e-3
- **Scale**: β × kl/p ≈ 0.15 for β=0.1, kl/p=1.5

**Watch for:**
1. **W_std should stay > 0.15** throughout training
   - If it drops below 0.10, β is too large
2. **KL/p should stabilize** around 1-2, not keep decreasing
   - Decreasing KL/p → posterior is collapsing
3. **β schedule**: Should reach full value slowly
   - With β=0.1, warmup=5k, reaches full β at step 5000

## Understanding β Values (with KL normalization)

**OLD (without normalization):**
| β Value | Effective |
|---------|-----------|
| 1e-4 | ~0.16 |
| 1e-5 | ~0.016 |

**NEW (with normalization):**
| β Value | Effect | Use When |
|---------|--------|----------|
| 0.5 | Very strong regularization | Large datasets, strong priors |
| 0.2 | Strong regularization | Standard case |
| 0.1 | **Moderate (RECOMMENDED)** | Start here |
| 0.05 | Weak regularization | Need more flexibility |
| 0.01 | Very weak | Almost deterministic |

## After Retraining

Run diagnostics to verify:

```bash
# 1. Check sampling works
python diagnose_posterior.py

# 2. Evaluate uncertainty
python evaluate_uncertainty.py \
    --dataset checkerboard \
    --checkpoint outputs/bfm/checkerboard/ckpt.pth \
    --n-posterior 50
```

**Target metrics:**
- ✓ Spearman ρ > 0.2 (positive correlation)
- ✓ Joint coverage within ±10% of expected
- ✓ W_std > 0.10 at end of training

## Understanding Joint vs Marginal Coverage

**Joint (Multivariate Ellipsoid)**: CORRECT test for d>1
- Tests if ground truth falls within α-credible ellipsoid
- Uses Mahalanobis distance: Δ = (v - μ)ᵀ Σ⁻¹ (v - μ)
- Threshold: Δ ≤ χ²_{d,α}
- Accounts for correlations between dimensions

**Marginal (Per-Dimension)**: INCORRECT but kept for comparison
- Tests if each dimension separately falls in α-credible interval
- Ignores correlations between dimensions
- Systematically under-reports coverage (anti-conservative)
- Box region ⊆ Ellipsoid region

**Expected behavior:**
- Marginal coverage ≤ Joint coverage
- Difference increases with stronger correlations
- Both should be close to expected α for well-calibrated model

## Advanced: Adaptive β

If you want to automatically tune β:

```python
# In training loop (experimental)
if kl < target_kl:
    beta *= 0.95  # Decrease β if KL too small
elif kl > target_kl * 2:
    beta *= 1.05  # Increase β if KL too large
```

Target KL depends on model size:
- For 2D output, ~100-500 is reasonable
- For larger models, ~1000-5000

## Why This Happens

The ELBO objective is:

```
L = MSE/(2σ²) + β·KL
```

- If β is too large, the optimizer focuses on minimizing KL
- Minimizing KL → posterior variance decreases
- Small posterior variance → overconfident predictions
- Overconfident → under-coverage and wrong uncertainty

## Expected Training Curves

**Good training:**
- MSE: Decreases steadily
- KL: Stays roughly constant or decreases slightly
- W_std: Stays > 0.10

**Bad training (what you had):**
- MSE: Decreases
- KL: Keeps decreasing (1600 → 710)
- W_std: Gets too small (< 0.10)

## Quick Diagnostic

Run this to check your current model:

```bash
python -c "
import torch
ckpt = torch.load('outputs/bfm/checkerboard/ckpt.pth', map_location='cpu')
W_std = torch.exp(ckpt['state_dict']['head.W_logsig']).mean()
print(f'Current W_std: {W_std:.4f}')
print('Status: ', end='')
if W_std < 0.05:
    print('✗ TOO SMALL - retrain with smaller β')
elif W_std < 0.10:
    print('⚠ BORDERLINE - try smaller β')
elif W_std < 0.30:
    print('✓ GOOD RANGE')
else:
    print('⚠ VERY LARGE - posterior might be too uncertain')
"
```

## Summary

1. **Retrain with β=1e-5** (10x smaller than your 1e-4)
2. **Use longer warmup** (5k steps instead of 2k)
3. **Monitor W_std** during training (should stay > 0.10)
4. **Check calibration** after training with evaluate_uncertainty.py

The goal is **positive ρ > 0.2** and **coverage within ±10%** of expected.
