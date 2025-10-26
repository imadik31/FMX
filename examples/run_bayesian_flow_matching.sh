#!/bin/bash

# Example workflow for Bayesian Flow Matching
# This script demonstrates the full pipeline from training to uncertainty analysis

set -e  # Exit on error

echo "========================================="
echo "Bayesian Flow Matching Example Workflow"
echo "========================================="
echo ""

# Configuration
DATASET="checkerboard"
OUTPUT_DIR="outputs"

echo "Dataset: $DATASET"
echo "Output directory: $OUTPUT_DIR"
echo ""

# ============================================
# 1. Train Bayesian Flow Matching
# ============================================
echo "========================================="
echo "Step 1: Training Bayesian Flow Matching"
echo "========================================="
echo ""

python train_bayesian_flow_matching_2d.py \
    --dataset $DATASET \
    --output-dir $OUTPUT_DIR \
    --beta 1e-4 \
    --sigma-likelihood 1.0 \
    --sigma-prior 1.0 \
    --beta-warmup-steps 2000 \
    --iterations 20000 \
    --hidden-dim 512 \
    --num-layers 3 \
    --n-mc-train 1

echo ""
echo "Training complete! Model saved to $OUTPUT_DIR/bfm/$DATASET/ckpt.pth"
echo ""

# ============================================
# 2. Analyze Uncertainty
# ============================================
echo "========================================="
echo "Step 2: Uncertainty Analysis"
echo "========================================="
echo ""

python sample_with_uncertainty.py \
    --dataset $DATASET \
    --checkpoint $OUTPUT_DIR/bfm/$DATASET/ckpt.pth \
    --output-dir $OUTPUT_DIR/uncertainty \
    --n-posterior 20 \
    --n-samples 1000 \
    --n-trajectories 5 \
    --hidden-dim 512 \
    --num-layers 3 \
    --sigma-prior 1.0

echo ""
echo "Uncertainty analysis complete!"
echo ""

# ============================================
# 3. (Optional) Train standard CFM for comparison
# ============================================
echo "========================================="
echo "Step 3: Training Standard CFM (for comparison)"
echo "========================================="
echo ""

python train_flow_matching_2d.py \
    --dataset $DATASET \
    --output-dir $OUTPUT_DIR

echo ""
echo "Standard CFM training complete!"
echo ""

# ============================================
# 4. (Optional) Laplace approximation on CFM model
# ============================================
echo "========================================="
echo "Step 4: Laplace Approximation (alternative approach)"
echo "========================================="
echo ""

python laplace_approximation.py \
    --dataset $DATASET \
    --checkpoint $OUTPUT_DIR/cfm/$DATASET/ckpt.pth \
    --output-dir $OUTPUT_DIR/laplace \
    --damping 1e-3 \
    --n-batches 50

echo ""
echo "Laplace approximation complete!"
echo ""

# ============================================
# Summary
# ============================================
echo "========================================="
echo "Workflow Complete!"
echo "========================================="
echo ""
echo "Generated outputs:"
echo ""
echo "Bayesian Flow Matching:"
echo "  - Model: $OUTPUT_DIR/bfm/$DATASET/ckpt.pth"
echo "  - Losses: $OUTPUT_DIR/bfm/$DATASET/losses.png"
echo "  - Sampling: $OUTPUT_DIR/bfm/$DATASET/sampling_*.png"
echo "  - Vector field: $OUTPUT_DIR/bfm/$DATASET/vector_field_*.gif"
echo ""
echo "Uncertainty Analysis:"
echo "  - Samples: $OUTPUT_DIR/uncertainty/$DATASET/uncertainty_samples.png"
echo "  - Trajectories: $OUTPUT_DIR/uncertainty/$DATASET/uncertainty_trajectories.png"
echo ""
echo "Standard CFM (for comparison):"
echo "  - Model: $OUTPUT_DIR/cfm/$DATASET/ckpt.pth"
echo "  - Losses: $OUTPUT_DIR/cfm/$DATASET/losses.png"
echo ""
echo "Laplace Approximation:"
echo "  - Parameters: $OUTPUT_DIR/laplace/$DATASET/laplace_params.pth"
echo ""
echo "========================================="
