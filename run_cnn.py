"""
Run CNN Pipeline
==================
End-to-end script: load data → train MobileNetV2 → evaluate → save.
Optimized for CPU training (no GPU required).
"""

import os
import sys
import logging

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import set_seed, setup_logging, ensure_dirs, get_device
from src.dataset import get_data_loaders
from src.cnn_model import get_model
from src.train_cnn import train_cnn, validate
from src.evaluate import (
    compute_metrics,
    format_metrics,
    plot_predicted_vs_actual,
    plot_residuals,
    plot_training_curves,
)


def main():
    setup_logging()
    set_seed(42)
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    ensure_dirs(project_root)
    
    csv_path = os.path.join(project_root, "data", "metadata.csv")
    
    if not os.path.exists(csv_path):
        logging.error(f"Dataset not found at {csv_path}. Run generate_synthetic_data.py first!")
        sys.exit(1)
    
    device = get_device()
    
    # ── Step 1: Create DataLoaders ────────────────────────────────
    logging.info("=" * 60)
    logging.info("STEP 1: Loading dataset and creating DataLoaders")
    logging.info("=" * 60)
    
    train_loader, val_loader, test_loader, dataset = get_data_loaders(
        csv_path=csv_path,
        root_dir=project_root,
        batch_size=32,
        val_split=0.15,
        test_split=0.15,
        image_size=224,
        seed=42,
    )
    
    logging.info(f"  Train batches: {len(train_loader)}")
    logging.info(f"  Val batches:   {len(val_loader)}")
    logging.info(f"  Test batches:  {len(test_loader)}")
    
    # ── Step 2: Train CNN ─────────────────────────────────────────
    logging.info("\n" + "=" * 60)
    logging.info("STEP 2: Training MobileNetV2 CNN")
    logging.info("=" * 60)
    
    save_dir = os.path.join(project_root, "models", "cnn")
    plot_dir = os.path.join(project_root, "results", "plots")
    
    # Tuned hyperparameters for accurate regression
    model, history = train_cnn(
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir=save_dir,
        phase1_epochs=30,
        phase2_epochs=20,
        phase1_lr=1e-3,
        phase2_lr=1e-4,
        unfreeze_layers=4,
        patience=10,
    )
    
    # ── Step 3: Plot Training Curves ──────────────────────────────
    logging.info("\nSTEP 3: Plotting training curves")
    
    plot_training_curves(
        history["train_loss"],
        history["val_loss"],
        phase_boundaries=history.get("phase_boundaries"),
        title="MobileNetV2 Training & Validation Loss",
        save_path=os.path.join(plot_dir, "cnn_training_curves.png"),
    )
    
    # ── Step 4: Evaluate on Test Set ──────────────────────────────
    logging.info("\n" + "=" * 60)
    logging.info("STEP 4: Evaluating on test set")
    logging.info("=" * 60)
    
    criterion = torch.nn.HuberLoss(delta=10.0)
    test_loss, y_pred, y_true = validate(model, test_loader, criterion, device)
    
    metrics = compute_metrics(y_true, y_pred)
    logging.info(format_metrics(metrics, "MobileNetV2 CNN"))
    
    # Predicted vs Actual plot
    plot_predicted_vs_actual(
        y_true, y_pred,
        title="MobileNetV2 CNN — Predicted vs Actual",
        save_path=os.path.join(plot_dir, "cnn_predicted_vs_actual.png"),
    )
    
    # Residual analysis
    plot_residuals(
        y_true, y_pred,
        title="MobileNetV2 CNN — Residual Analysis",
        save_path=os.path.join(plot_dir, "cnn_residuals.png"),
    )
    
    # Save metrics
    import json
    metrics_path = os.path.join(project_root, "results", "cnn_metrics.json")
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logging.info(f"Metrics saved to {metrics_path}")
    
    logging.info("\n" + "=" * 60)
    logging.info("CNN pipeline complete! Check results/plots/ for visualizations.")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
