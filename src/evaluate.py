"""
Evaluation & Visualization Module
====================================
Compute regression metrics and generate publication-quality plots
for model comparison and result analysis.
"""

import os
import logging

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


# Set global plot style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

PLOT_DEFAULTS = {
    "figure.figsize": (10, 7),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "lines.linewidth": 2,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
}
plt.rcParams.update(PLOT_DEFAULTS)


def compute_metrics(y_true, y_pred) -> dict:
    """
    Compute comprehensive regression metrics.
    
    Returns dict with: RMSE, MAE, R², MAPE, max_error
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    nonzero = np.abs(y_true) > 1e-6

    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100) if nonzero.any() else float("nan"),
        "max_error": float(np.max(np.abs(y_true - y_pred))),
        "n_samples": len(y_true),
    }
    
    return metrics


def format_metrics(metrics: dict, model_name: str = "Model") -> str:
    """Format metrics dict as a readable string."""
    return (
        f"\n{'─' * 45}\n"
        f"  {model_name} — Evaluation Results\n"
        f"{'─' * 45}\n"
        f"  RMSE      : {metrics['rmse']:.4f} ppm\n"
        f"  MAE       : {metrics['mae']:.4f} ppm\n"
        f"  R²        : {metrics['r2']:.4f}\n"
        f"  MAPE      : {metrics['mape']:.2f}%\n"
        f"  Max Error : {metrics['max_error']:.4f} ppm\n"
        f"  Samples   : {metrics['n_samples']}\n"
        f"{'─' * 45}"
    )


def plot_predicted_vs_actual(
    y_true, y_pred,
    title: str = "Predicted vs Actual Concentration",
    save_path: str = None,
):
    """
    Scatter plot of predicted vs actual ppm values with ideal line.
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    metrics = compute_metrics(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Scatter plot
    scatter = ax.scatter(
        y_true, y_pred,
        alpha=0.5, s=30, c=y_true,
        cmap="viridis", edgecolors="white", linewidth=0.3,
    )
    plt.colorbar(scatter, ax=ax, label="Actual ppm", shrink=0.8)
    
    # Ideal line (y = x)
    limits = [
        min(y_true.min(), y_pred.min()) - 5,
        max(y_true.max(), y_pred.max()) + 5,
    ]
    ax.plot(limits, limits, "r--", linewidth=2, alpha=0.7, label="Ideal (y = x)")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    
    # Labels and title
    ax.set_xlabel("Actual Concentration (ppm)")
    ax.set_ylabel("Predicted Concentration (ppm)")
    ax.set_title(title)
    
    # Metrics annotation
    textstr = (f"R² = {metrics['r2']:.4f}\n"
               f"RMSE = {metrics['rmse']:.3f} ppm\n"
               f"MAE = {metrics['mae']:.3f} ppm")
    props = dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment="top", bbox=props)
    
    ax.legend(loc="lower right")
    ax.set_aspect("equal")
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
        logging.info(f"Plot saved: {save_path}")
    
    plt.close(fig)
    return fig


def plot_residuals(
    y_true, y_pred,
    title: str = "Residual Analysis",
    save_path: str = None,
):
    """
    Two-panel residual analysis: residuals vs predicted + distribution histogram.
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel 1: Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.5, s=25, color="steelblue", edgecolors="white", linewidth=0.3)
    axes[0].axhline(y=0, color="red", linestyle="--", linewidth=1.5)
    axes[0].set_xlabel("Predicted Concentration (ppm)")
    axes[0].set_ylabel("Residual (Actual - Predicted)")
    axes[0].set_title("Residuals vs Predicted")
    
    # Panel 2: Residual Distribution
    axes[1].hist(residuals, bins=30, color="steelblue", edgecolor="white", alpha=0.8, density=True)
    
    # Overlay normal distribution
    mu, sigma = residuals.mean(), residuals.std()
    x = np.linspace(residuals.min(), residuals.max(), 100)
    axes[1].plot(x, (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2),
                  color="red", linewidth=2, label=f"Normal(μ={mu:.2f}, σ={sigma:.2f})")
    axes[1].set_xlabel("Residual (ppm)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Residual Distribution")
    axes[1].legend()
    
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
        logging.info(f"Plot saved: {save_path}")
    
    plt.close(fig)
    return fig


def plot_training_curves(
    train_losses,
    val_losses,
    phase_boundaries=None,
    title: str = "Training & Validation Loss",
    save_path: str = None,
):
    """
    Plot training and validation loss curves with optional phase boundaries.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    ax.plot(epochs, train_losses, "b-", linewidth=2, label="Training Loss", alpha=0.8)
    ax.plot(epochs, val_losses, "r-", linewidth=2, label="Validation Loss", alpha=0.8)
    
    # Mark phase boundaries
    if phase_boundaries:
        for i, boundary in enumerate(phase_boundaries[:-1]):
            ax.axvline(x=boundary, color="green", linestyle="--", alpha=0.7,
                        label=f"Phase {i + 2} start" if i == 0 else "")
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title(title)
    ax.legend()
    ax.set_yscale("log")
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
        logging.info(f"Plot saved: {save_path}")
    
    plt.close(fig)
    return fig


def compare_models(
    results: dict,
    save_path: str = None,
):
    """
    Bar chart comparing multiple models across metrics.
    
    Args:
        results: dict of {model_name: metrics_dict}
        save_path: Optional path to save the plot
    """
    model_names = list(results.keys())
    metrics_to_plot = ["rmse", "mae", "r2"]
    metric_labels = ["RMSE (ppm)", "MAE (ppm)", "R²"]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    colors = sns.color_palette("husl", len(model_names))
    
    for i, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
        values = [results[name][metric] for name in model_names]
        bars = axes[i].bar(model_names, values, color=colors, edgecolor="white", linewidth=1.5)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            axes[i].text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * max(values),
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold",
            )
        
        axes[i].set_ylabel(label)
        axes[i].set_title(label)
        axes[i].tick_params(axis="x", rotation=30)
    
    fig.suptitle("Model Comparison", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
        logging.info(f"Plot saved: {save_path}")
    
    plt.close(fig)
    return fig


def plot_feature_importance(
    feature_names,
    importances,
    title: str = "Feature Importance (Random Forest)",
    save_path: str = None,
):
    """Plot feature importance as horizontal bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by importance
    sorted_idx = np.argsort(importances)
    
    ax.barh(
        range(len(sorted_idx)),
        importances[sorted_idx],
        color="steelblue",
        edgecolor="white",
    )
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax.set_xlabel("Feature Importance")
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
        logging.info(f"Plot saved: {save_path}")
    
    plt.close(fig)
    return fig
