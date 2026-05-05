"""
Run Baseline Models Pipeline
===============================
End-to-end script: load data → extract features → train baselines → evaluate → save.
"""

import os
import sys
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import set_seed, setup_logging, ensure_dirs
from src.baseline_models import (
    extract_features_from_dataset,
    train_linear_regression,
    train_random_forest,
    train_svr,
    train_gradient_boosting,
    save_model,
    FEATURE_COLUMNS,
)
from src.evaluate import (
    compute_metrics,
    format_metrics,
    plot_predicted_vs_actual,
    plot_residuals,
    compare_models,
    plot_feature_importance,
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
    
    # ── Step 1: Extract Features ──────────────────────────────────
    logging.info("=" * 60)
    logging.info("STEP 1: Extracting color features from images")
    logging.info("=" * 60)
    
    feature_df = extract_features_from_dataset(csv_path, root_dir=project_root)
    
    # Save features for reuse
    features_csv = os.path.join(project_root, "data", "features.csv")
    feature_df.to_csv(features_csv, index=False)
    logging.info(f"Features saved to {features_csv}")
    
    # ── Step 2: Prepare Train/Test Split ─────────────────────────
    logging.info("\nSTEP 2: Preparing train/test split")
    
    X = feature_df[FEATURE_COLUMNS].values
    y = feature_df["ppm"].values
    
    # Stratified split by dye_type to ensure balanced representation
    stratify_col = feature_df["dye_type"].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_col,
    )
    
    logging.info(f"  Train: {X_train.shape[0]} samples")
    logging.info(f"  Test:  {X_test.shape[0]} samples")
    logging.info(f"  Features: {X_train.shape[1]} (incl. dye_type one-hot)")
    
    # ── Step 3: Train Models ─────────────────────────────────────
    logging.info("\n" + "=" * 60)
    logging.info("STEP 3: Training baseline models")
    logging.info("=" * 60)
    
    all_results = {}
    model_dir = os.path.join(project_root, "models", "baseline")
    plot_dir = os.path.join(project_root, "results", "plots")
    
    # 3a. Linear Regression
    logging.info("\n── Linear Regression ──")
    lr_model, lr_pred, lr_meta = train_linear_regression(X_train, y_train, X_test, y_test)
    lr_metrics = compute_metrics(y_test, lr_pred)
    all_results["Linear Regression"] = lr_metrics
    save_model(lr_model, os.path.join(model_dir, "linear_regression.joblib"))
    logging.info(format_metrics(lr_metrics, "Linear Regression"))
    
    plot_predicted_vs_actual(
        y_test, lr_pred,
        title="Linear Regression — Predicted vs Actual",
        save_path=os.path.join(plot_dir, "lr_predicted_vs_actual.png"),
    )
    
    # 3b. Random Forest
    logging.info("\n── Random Forest ──")
    rf_model, rf_pred, rf_meta = train_random_forest(X_train, y_train, X_test, y_test)
    rf_metrics = compute_metrics(y_test, rf_pred)
    all_results["Random Forest"] = rf_metrics
    save_model(rf_model, os.path.join(model_dir, "random_forest.joblib"))
    logging.info(format_metrics(rf_metrics, "Random Forest"))
    
    plot_predicted_vs_actual(
        y_test, rf_pred,
        title="Random Forest — Predicted vs Actual",
        save_path=os.path.join(plot_dir, "rf_predicted_vs_actual.png"),
    )
    
    # Feature importance plot for RF
    rf_importances = rf_model.named_steps["model"].feature_importances_
    plot_feature_importance(
        FEATURE_COLUMNS,
        rf_importances,
        title="Random Forest — Feature Importance",
        save_path=os.path.join(plot_dir, "rf_feature_importance.png"),
    )
    
    # 3c. SVR
    logging.info("\n── Support Vector Regression ──")
    svr_model, svr_pred, svr_meta = train_svr(X_train, y_train, X_test, y_test)
    svr_metrics = compute_metrics(y_test, svr_pred)
    all_results["SVR"] = svr_metrics
    save_model(svr_model, os.path.join(model_dir, "svr.joblib"))
    logging.info(format_metrics(svr_metrics, "SVR"))
    
    plot_predicted_vs_actual(
        y_test, svr_pred,
        title="SVR — Predicted vs Actual",
        save_path=os.path.join(plot_dir, "svr_predicted_vs_actual.png"),
    )
    
    # 3d. Gradient Boosting
    logging.info("\n── Gradient Boosting ──")
    gb_model, gb_pred, gb_meta = train_gradient_boosting(X_train, y_train, X_test, y_test)
    gb_metrics = compute_metrics(y_test, gb_pred)
    all_results["Gradient Boosting"] = gb_metrics
    save_model(gb_model, os.path.join(model_dir, "gradient_boosting.joblib"))
    logging.info(format_metrics(gb_metrics, "Gradient Boosting"))
    
    plot_predicted_vs_actual(
        y_test, gb_pred,
        title="Gradient Boosting — Predicted vs Actual",
        save_path=os.path.join(plot_dir, "gb_predicted_vs_actual.png"),
    )
    
    # ── Step 4: Model Comparison ──────────────────────────────────
    logging.info("\n" + "=" * 60)
    logging.info("STEP 4: Model Comparison")
    logging.info("=" * 60)
    
    compare_models(
        all_results,
        save_path=os.path.join(plot_dir, "baseline_model_comparison.png"),
    )
    
    # Residual analysis for best model
    best_model_name = max(all_results, key=lambda k: all_results[k]["r2"])
    logging.info(f"\nBest baseline model: {best_model_name} (R² = {all_results[best_model_name]['r2']:.4f})")
    
    best_preds = {"Linear Regression": lr_pred, "Random Forest": rf_pred,
                   "SVR": svr_pred, "Gradient Boosting": gb_pred}
    
    plot_residuals(
        y_test, best_preds[best_model_name],
        title=f"{best_model_name} — Residual Analysis",
        save_path=os.path.join(plot_dir, "best_baseline_residuals.png"),
    )
    
    # Save summary
    summary_df = pd.DataFrame(all_results).T
    summary_df.index.name = "model"
    summary_path = os.path.join(project_root, "results", "baseline_summary.csv")
    summary_df.to_csv(summary_path)
    logging.info(f"\nSummary saved to {summary_path}")
    logging.info(f"\n{summary_df.to_string()}")
    
    logging.info("\n" + "=" * 60)
    logging.info("Baseline pipeline complete! Check results/plots/ for visualizations.")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
