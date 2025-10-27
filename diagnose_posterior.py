"""
Diagnostic script to check if posterior sampling is working correctly.

This script tests:
1. Are posterior samples actually different?
2. Is the posterior variance reasonable?
3. Is the KL term working correctly?
"""

import torch
from flow_matching.datasets import TOY_DATASETS
from flow_matching.models.bayesian import BayesianMLP
from flow_matching.utils import set_seed

def test_posterior_sampling():
    """Test if posterior sampling produces different values."""
    print("=" * 80)
    print("Testing Posterior Sampling")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    # Load checkpoint
    checkpoint = torch.load("outputs/bfm/checkerboard/ckpt.pth", map_location=device)

    if isinstance(checkpoint, dict) and 'config' in checkpoint:
        config = checkpoint['config']
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        # Infer config
        backbone_keys = [k for k in state_dict.keys() if k.startswith('backbone.') and 'weight' in k]
        max_idx = max([int(k.split('.')[1]) for k in backbone_keys])
        config = {
            'dim': 2,
            'time_dim': 1,
            'hidden_dim': state_dict['backbone.0.weight'].shape[0],
            'num_layers': (max_idx + 1) // 2,
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
    print(f"Model loaded: hidden_dim={config['hidden_dim']}, num_layers={config['num_layers']}")

    # Check posterior parameters
    print("\n" + "=" * 80)
    print("Posterior Parameters")
    print("=" * 80)
    with torch.no_grad():
        W_mu = model.head.W_mu
        b_mu = model.head.b_mu
        W_std = torch.exp(model.head.W_logsig)
        b_std = torch.exp(model.head.b_logsig)

        print(f"W_mu: shape={W_mu.shape}, mean={W_mu.abs().mean():.6f}, std={W_mu.std():.6f}")
        print(f"W_std: shape={W_std.shape}, mean={W_std.mean():.6f}, min={W_std.min():.6f}, max={W_std.max():.6f}")
        print(f"b_mu: shape={b_mu.shape}, mean={b_mu.abs().mean():.6f}, std={b_mu.std():.6f}")
        print(f"b_std: shape={b_std.shape}, mean={b_std.mean():.6f}, min={b_std.min():.6f}, max={b_std.max():.6f}")

        # Check signal-to-noise ratio
        W_snr = (W_mu.abs().mean() / W_std.mean()).item()
        b_snr = (b_mu.abs().mean() / b_std.mean()).item()
        print(f"\nSignal-to-noise ratio:")
        print(f"  W: {W_snr:.2f}  (higher means posterior is more confident)")
        print(f"  b: {b_snr:.2f}")

        if W_snr > 10:
            print("  ⚠ WARNING: SNR too high! Posterior is overconfident.")
        elif W_snr < 0.1:
            print("  ⚠ WARNING: SNR too low! Posterior is too uncertain.")
        else:
            print("  ✓ SNR looks reasonable")

    # Test sampling with a single input
    print("\n" + "=" * 80)
    print("Testing Sampling Diversity")
    print("=" * 80)

    dataset = TOY_DATASETS['checkerboard'](device=device)
    x_1 = dataset.sample(1)
    x_0 = torch.randn_like(x_1)
    t = torch.tensor([[0.5]], device=device)
    x_t = (1 - t) * x_0 + t * x_1
    v_gt = x_1 - x_0

    print(f"Test point: x_t={x_t[0].cpu().numpy()}, t={t[0].item()}")
    print(f"Ground truth velocity: v_gt={v_gt[0].cpu().numpy()}")

    # Collect 100 posterior samples
    n_samples = 100
    samples = []
    with torch.no_grad():
        for i in range(n_samples):
            v_pred = model.forward_sample(x_t, t, n_mc=1).squeeze(0)  # [1, 2]
            samples.append(v_pred)

    samples = torch.stack(samples, dim=0)  # [100, 1, 2]
    samples = samples.squeeze(1)  # [100, 2]

    # Compute statistics
    mean = samples.mean(dim=0)
    std = samples.std(dim=0)

    print(f"\nPosterior statistics (100 samples):")
    print(f"  Mean: {mean.cpu().numpy()}")
    print(f"  Std:  {std.cpu().numpy()}")
    print(f"  Error (mean - gt): {(mean - v_gt[0]).cpu().numpy()}")
    print(f"  RMSE: {(mean - v_gt[0]).pow(2).sum().sqrt().item():.6f}")

    # Check if samples are actually different
    unique_samples = torch.unique(samples, dim=0).size(0)
    print(f"\n  Unique samples: {unique_samples}/{n_samples}")

    if unique_samples < n_samples * 0.9:
        print("  ✗ WARNING: Many duplicate samples! Sampling may not be working.")
    else:
        print("  ✓ Samples are diverse")

    # Check range of samples
    sample_range = samples.max(dim=0)[0] - samples.min(dim=0)[0]
    print(f"  Sample range: {sample_range.cpu().numpy()}")

    if sample_range.max() < 0.01:
        print("  ✗ WARNING: Very small sample range! Posterior is too confident.")
    else:
        print("  ✓ Sample range looks reasonable")

    # Visualize distribution
    print(f"\n  First 10 samples:")
    for i in range(10):
        print(f"    {i}: {samples[i].cpu().numpy()}")

    # Check if std correlates with uncertainty
    print("\n" + "=" * 80)
    print("Testing on Multiple Points")
    print("=" * 80)

    n_test = 100
    x_1_test = dataset.sample(n_test)
    x_0_test = torch.randn_like(x_1_test)
    t_test = torch.rand(n_test, 1, device=device)
    x_t_test = (1 - t_test) * x_0_test + t_test * x_1_test
    v_gt_test = x_1_test - x_0_test

    # Collect samples for each point
    n_posterior = 50
    all_preds = []
    with torch.no_grad():
        for _ in range(n_posterior):
            v_pred = model.forward_sample(x_t_test, t_test, n_mc=1).squeeze(0)
            all_preds.append(v_pred)

    all_preds = torch.stack(all_preds, dim=0)  # [50, 100, 2]
    mean_pred = all_preds.mean(dim=0)  # [100, 2]
    std_pred = all_preds.std(dim=0)  # [100, 2]

    uncertainty_mag = std_pred.norm(dim=-1)  # [100]
    error_mag = (mean_pred - v_gt_test).pow(2).sum(dim=-1).sqrt()  # [100]

    # Compute correlation
    from scipy.stats import spearmanr
    rho, p_value = spearmanr(uncertainty_mag.cpu().numpy(), error_mag.cpu().numpy())

    print(f"  Mean uncertainty: {uncertainty_mag.mean().item():.6f}")
    print(f"  Mean error: {error_mag.mean().item():.6f}")
    print(f"  Spearman correlation: {rho:.4f} (p={p_value:.2e})")

    if rho < 0:
        print("  ✗ NEGATIVE CORRELATION: Model is confident where it's wrong!")
    elif rho > 0.3:
        print("  ✓ Good positive correlation")
    else:
        print("  ⚠ Weak positive correlation")

    # Check hidden features
    print("\n" + "=" * 80)
    print("Analyzing Hidden Features")
    print("=" * 80)

    with torch.no_grad():
        feats = model.encode(x_t_test[:10], t_test[:10])  # [10, hidden_dim]
        print(f"  Hidden features shape: {feats.shape}")
        print(f"  Mean norm: {feats.norm(dim=-1).mean().item():.6f}")
        print(f"  Min norm: {feats.norm(dim=-1).min().item():.6f}")
        print(f"  Max norm: {feats.norm(dim=-1).max().item():.6f}")
        print(f"  Mean value: {feats.mean().item():.6f}")
        print(f"  Std value: {feats.std().item():.6f}")

        # Expected output variance
        expected_output_std = feats.norm(dim=-1).mean() * W_std.mean()
        print(f"\n  Expected output std (||h|| * σ_W): {expected_output_std.item():.6f}")
        print(f"  Observed output std: {std_pred.mean().item():.6f}")

        if observed_std := std_pred.mean().item():
            ratio = expected_output_std.item() / observed_std
            if ratio < 0.5 or ratio > 2.0:
                print(f"  ⚠ WARNING: Ratio {ratio:.2f} suggests issue with variance propagation")
            else:
                print(f"  ✓ Ratio {ratio:.2f} looks reasonable")

    print("\n" + "=" * 80)
    print("Diagnosis Complete")
    print("=" * 80)


if __name__ == "__main__":
    test_posterior_sampling()
