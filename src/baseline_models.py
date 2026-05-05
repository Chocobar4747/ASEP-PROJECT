"""
Baseline Regression Models
============================
Train classical ML models (Linear Regression, Random Forest, SVR)
on extracted RGB/HSV color features for dye concentration prediction.
"""

import os
import logging
import warnings

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline

from src.preprocessing import load_image, crop_roi, resize_image, extract_color_features

warnings.filterwarnings("ignore", category=FutureWarning)


# ── Feature names used for training ──
# Core color features
COLOR_FEATURE_COLUMNS = [
    "r_mean", "r_std", "g_mean", "g_std", "b_mean", "b_std",
    "h_mean", "h_std", "s_mean", "s_std", "v_mean", "v_std",
    "intensity", "r_ratio", "g_ratio", "b_ratio",
    "dominant_r", "dominant_g", "dominant_b",
]

# Dye type one-hot encoded columns (added for disambiguation)
DYE_TYPE_COLUMNS = [
    "dye_methylene_blue", "dye_congo_red", "dye_crystal_violet",
]

# Full feature set
FEATURE_COLUMNS = COLOR_FEATURE_COLUMNS + DYE_TYPE_COLUMNS


def extract_features_from_dataset(
    csv_path: str,
    root_dir: str = ".",
    max_samples: int = None,
) -> pd.DataFrame:
    """
    Extract color features from all images in the dataset.
    
    Args:
        csv_path: Path to metadata.csv
        root_dir: Project root directory
        max_samples: Optional limit on number of samples to process
    
    Returns:
        DataFrame with features + ppm + dye_type columns
    """
    df = pd.read_csv(csv_path)
    if max_samples:
        df = df.head(max_samples)
    
    logging.info(f"Extracting features from {len(df)} images...")
    
    all_features = []
    for idx, row in df.iterrows():
        img_path = os.path.join(root_dir, row["image_path"])
        
        try:
            image = load_image(img_path)
            image = crop_roi(image, method="center", crop_fraction=0.7)
            image = resize_image(image, size=(224, 224))
            features = extract_color_features(image)
            features["ppm"] = row["ppm"]
            features["dye_type"] = row["dye_type"]
            
            # One-hot encode dye type for disambiguation
            features["dye_methylene_blue"] = 1.0 if row["dye_type"] == "methylene_blue" else 0.0
            features["dye_congo_red"] = 1.0 if row["dye_type"] == "congo_red" else 0.0
            features["dye_crystal_violet"] = 1.0 if row["dye_type"] == "crystal_violet" else 0.0
            
            all_features.append(features)
        except Exception as e:
            logging.warning(f"Failed to process {img_path}: {e}")
            continue
        
        if (idx + 1) % 200 == 0:
            logging.info(f"  Processed {idx + 1}/{len(df)}")
    
    feature_df = pd.DataFrame(all_features)
    logging.info(f"Feature extraction complete: {len(feature_df)} samples, "
                  f"{len(FEATURE_COLUMNS)} features each")
    
    return feature_df


def train_linear_regression(X_train, y_train, X_test, y_test):
    """
    Train Linear Regression with Ridge regularization.
    
    Returns: (model_pipeline, predictions, metrics_dict)
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ])
    
    # Cross-validation score
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5,
                                  scoring="neg_root_mean_squared_error")
    
    # Train on full training set
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    metrics = {
        "cv_rmse_mean": -cv_scores.mean(),
        "cv_rmse_std": cv_scores.std(),
    }
    
    logging.info(f"Linear Regression — CV RMSE: {metrics['cv_rmse_mean']:.3f} "
                  f"± {metrics['cv_rmse_std']:.3f}")
    
    return pipeline, y_pred, metrics


def train_random_forest(X_train, y_train, X_test, y_test):
    """
    Train Random Forest Regressor with hyperparameter tuning.
    
    Returns: (model_pipeline, predictions, metrics_dict)
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(random_state=42)),
    ])
    
    # Expanded hyperparameter grid for better tuning
    param_grid = {
        "model__n_estimators": [200, 300, 500],
        "model__max_depth": [10, 20, 30, None],
        "model__min_samples_split": [2, 3, 5],
        "model__min_samples_leaf": [1, 2],
    }
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=0,
    )
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    metrics = {
        "best_params": grid_search.best_params_,
        "best_cv_rmse": -grid_search.best_score_,
    }
    
    logging.info(f"Random Forest — Best CV RMSE: {metrics['best_cv_rmse']:.3f}")
    logging.info(f"  Best params: {metrics['best_params']}")
    
    return best_model, y_pred, metrics


def train_svr(X_train, y_train, X_test, y_test):
    """
    Train Support Vector Regression.
    
    Returns: (model_pipeline, predictions, metrics_dict)
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVR(kernel="rbf")),
    ])
    
    # Expanded hyperparameter grid
    param_grid = {
        "model__C": [1, 10, 50, 100],
        "model__epsilon": [0.01, 0.1, 0.5, 1.0],
        "model__gamma": ["scale", "auto"],
    }
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=0,
    )
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    metrics = {
        "best_params": grid_search.best_params_,
        "best_cv_rmse": -grid_search.best_score_,
    }
    
    logging.info(f"SVR — Best CV RMSE: {metrics['best_cv_rmse']:.3f}")
    logging.info(f"  Best params: {metrics['best_params']}")
    
    return best_model, y_pred, metrics


def train_gradient_boosting(X_train, y_train, X_test, y_test):
    """
    Train Gradient Boosting Regressor.
    
    Returns: (model_pipeline, predictions, metrics_dict)
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingRegressor(random_state=42)),
    ])
    
    # Expanded hyperparameter grid
    param_grid = {
        "model__n_estimators": [200, 300, 500],
        "model__max_depth": [3, 5, 7],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__subsample": [0.8, 1.0],
    }
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=0,
    )
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    metrics = {
        "best_params": grid_search.best_params_,
        "best_cv_rmse": -grid_search.best_score_,
    }
    
    logging.info(f"Gradient Boosting — Best CV RMSE: {metrics['best_cv_rmse']:.3f}")
    
    return best_model, y_pred, metrics


def save_model(model, filepath: str):
    """Save trained sklearn model to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    logging.info(f"Model saved to {filepath}")


def load_model(filepath: str):
    """Load trained sklearn model from disk."""
    return joblib.load(filepath)
