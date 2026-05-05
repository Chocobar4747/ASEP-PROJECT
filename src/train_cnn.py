"""
CNN Training Pipeline
=======================
Two-phase training strategy for the ResNet18 regression model:
  Phase 1: Frozen backbone, train regression head only (fast convergence)
  Phase 2: Unfreeze last N backbone layers, fine-tune end-to-end (accuracy boost)
"""

import os
import time
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from src.cnn_model import get_model


class EarlyStopping:
    """Early stopping to terminate training when validation loss stops improving."""
    
    def __init__(self, patience: int = 7, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logging.info(f"Early stopping triggered after {self.patience} epochs without improvement")
        else:
            self.best_loss = val_loss
            self.counter = 0


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for a single epoch. Returns average loss. Model outputs raw ppm."""
    model.train()
    running_loss = 0.0
    num_batches = 0
    
    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)  # raw ppm values
        
        optimizer.zero_grad()
        predictions = model(images)
        loss = criterion(predictions, targets)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item()
        num_batches += 1
    
    return running_loss / max(num_batches, 1)



@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """Validate model. Returns average loss and predictions in ppm scale."""
    model.eval()
    running_loss = 0.0
    num_batches = 0
    all_preds = []
    all_targets = []
    
    for images, targets in dataloader:
        images = images.to(device)
        targets_dev = targets.to(device)
        
        predictions = model(images)
        loss = criterion(predictions, targets_dev)
        
        running_loss += loss.item()
        num_batches += 1
        
        # Model outputs raw ppm — no denormalization needed
        all_preds.extend(predictions.cpu().numpy().flatten())
        all_targets.extend(targets.numpy().flatten())
    
    avg_loss = running_loss / max(num_batches, 1)
    return avg_loss, np.array(all_preds).flatten(), np.array(all_targets).flatten()


def train_cnn(
    train_loader,
    val_loader,
    device,
    save_dir: str = "models/cnn",
    phase1_epochs: int = 15,
    phase2_epochs: int = 20,
    phase1_lr: float = 1e-3,
    phase2_lr: float = 1e-4,
    unfreeze_layers: int = 4,
    patience: int = 7,
):
    """
    Full two-phase CNN training pipeline.
    
    Phase 1: Frozen backbone, train head only
    Phase 2: Unfreeze last N layers, fine-tune with small lr
    
    Args:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        device: torch device
        save_dir: Directory to save model checkpoints
        phase1_epochs: Number of epochs for Phase 1
        phase2_epochs: Number of epochs for Phase 2
        phase1_lr: Learning rate for Phase 1
        phase2_lr: Learning rate for Phase 2
        unfreeze_layers: Number of backbone layers to unfreeze in Phase 2
        patience: Early stopping patience
    
    Returns:
        (model, training_history dict)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize model
    model = get_model(pretrained=True, freeze_backbone=True)
    model = model.to(device)
    
    # Huber loss — more robust to outliers than MSE
    criterion = nn.HuberLoss(delta=10.0)
    history = {
        "train_loss": [], "val_loss": [],
        "phase_boundaries": [], "best_epoch": 0,
    }
    
    best_val_loss = float("inf")
    best_model_path = os.path.join(save_dir, "best_model.pth")
    
    # ════════════════════════════════════════════════════════════════
    # PHASE 1: Train regression head only (backbone frozen)
    # ════════════════════════════════════════════════════════════════
    logging.info("=" * 60)
    logging.info("PHASE 1: Training regression head (backbone frozen)")
    logging.info("=" * 60)
    
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=phase1_lr,
        weight_decay=1e-4,
    )
    
    # Warmup + Cosine annealing schedule
    warmup_epochs = min(3, phase1_epochs // 3)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=phase1_epochs - warmup_epochs, eta_min=phase1_lr * 0.01)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
    
    early_stopping = EarlyStopping(patience=patience)
    
    for epoch in range(1, phase1_epochs + 1):
        start_time = time.time()
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, _, _ = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        elapsed = time.time() - start_time
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        
        logging.info(
            f"[Phase 1] Epoch {epoch:3d}/{phase1_epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f} | Time: {elapsed:.1f}s"
        )
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            history["best_epoch"] = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "phase": 1,
            }, best_model_path)
        
        early_stopping(val_loss)
        if early_stopping.should_stop:
            break
    
    history["phase_boundaries"].append(len(history["train_loss"]))
    
    # ════════════════════════════════════════════════════════════════
    # PHASE 2: Fine-tune with unfrozen backbone layers
    # ════════════════════════════════════════════════════════════════
    logging.info("=" * 60)
    logging.info(f"PHASE 2: Fine-tuning (unfreezing last {unfreeze_layers} backbone layers)")
    logging.info("=" * 60)
    
    # Load best Phase 1 model
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Unfreeze last N layers
    model.unfreeze_last_n_layers(n=unfreeze_layers)
    params = model.count_parameters()
    logging.info(f"Trainable parameters after unfreezing: {params['trainable']:,}")
    
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=phase2_lr,
        weight_decay=1e-4,
    )
    
    warmup_epochs_p2 = min(2, phase2_epochs // 4)
    warmup_scheduler_p2 = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs_p2)
    cosine_scheduler_p2 = CosineAnnealingLR(optimizer, T_max=phase2_epochs - warmup_epochs_p2, eta_min=phase2_lr * 0.01)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler_p2, cosine_scheduler_p2], milestones=[warmup_epochs_p2])
    
    early_stopping = EarlyStopping(patience=patience)
    
    for epoch in range(1, phase2_epochs + 1):
        start_time = time.time()
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, _, _ = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        elapsed = time.time() - start_time
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        
        logging.info(
            f"[Phase 2] Epoch {epoch:3d}/{phase2_epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f} | Time: {elapsed:.1f}s"
        )
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            history["best_epoch"] = len(history["train_loss"])
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "phase": 2,
            }, best_model_path)
        
        early_stopping(val_loss)
        if early_stopping.should_stop:
            break
    
    history["phase_boundaries"].append(len(history["train_loss"]))
    
    # Load best model
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    logging.info("=" * 60)
    logging.info(f"Training complete! Best val loss: {best_val_loss:.4f} "
                  f"at epoch {history['best_epoch']}")
    logging.info(f"Best model saved to: {best_model_path}")
    logging.info("=" * 60)
    
    return model, history
