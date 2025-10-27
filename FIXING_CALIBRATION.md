# Fixing Under-Calibrated Uncertainty in BFM

## Problem Diagnosis

Your results show severe under-calibration:
- **Spearman ρ = -0.50**: Negative correlation (uncertainty anti-correlated with error!)
- **Coverage ~3-10%** instead of 50-95%: Posterior is way too confident
- **KL decreased** from 1600 → 710: Posterior collapsed during training

## Root Causes

1. **β is too large** (1e-4), causing over-regularization. The posterior variance was squeezed too tight during training.
2. **Coverage test was flawed** (FIXED): Previous test used per-dimension marginal intervals which systematically under-report coverage. Now uses correct joint multivariate ellipsoid test.

**Note**: After fixing the coverage test, your coverage numbers will be HIGHER. But the negative Spearman ρ still indicates β needs to be reduced.

## Quick Fixes

### Fix 1: Reduce β (RECOMMENDED)

```bash
python train_bayesian_flow_matching_2d.py \
    --dataset checkerboard \
    --beta 1e-5 \              # 10x smaller!
    --iterations 20000 \
    --beta-warmup-steps 5000   # Longer warmup
```

**Expected improvements:**
- Positive ρ (0.2-0.4)
- Coverage closer to expected (within ±10%)
- Higher posterior std

### Fix 2: Even Smaller β

If Fix 1 doesn't work, try:

```bash
python train_bayesian_flow_matching_2d.py \
    --dataset checkerboard \
    --beta 1e-6 \              # Very small
    --iterations 20000 \
    --beta-warmup-steps 10000  # Long warmup
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

With the updated training script, you'll see:

```
| step:   2000 | loss:   2.28 | mse:   4.25 | kl:  1600.67 | beta: 0.000010 | W_std: 0.1500 | b_std: 0.0100 |
```

**Watch for:**
1. **W_std should stay > 0.10** throughout training
   - If it drops below 0.05, β is too large
2. **KL should stabilize**, not keep decreasing
   - Decreasing KL → posterior is collapsing
3. **β schedule**: Should reach full value slowly
   - With β=1e-5, warmup=5k, reaches full β at step 5000

## Understanding β Values

| β Value | Effect | Use When |
|---------|--------|----------|
| 1e-3 | Very strong regularization | Large datasets, want strong priors |
| 1e-4 | Strong regularization | Default, but often too strong |
| 1e-5 | **Moderate (RECOMMENDED)** | Most cases, good starting point |
| 1e-6 | Weak regularization | Small datasets, need flexibility |
| 1e-7 | Very weak | Almost like deterministic training |

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
