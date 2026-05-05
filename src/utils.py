"""
Utility functions for the Dye Concentration Prediction system.
Handles reproducibility, device selection, logging, and directory setup.
"""

import os
import random
import logging
import numpy as np
import torch


def set_seed(seed: int = 42):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic mode for reproducibility (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Auto-detect best available device (CUDA → CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU (no GPU detected)")
    return device


def setup_logging(level=logging.INFO):
    """Configure consistent logging format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def ensure_dirs(base_dir: str = "."):
    """Create all required output directories."""
    dirs = [
        os.path.join(base_dir, "data", "raw", "methylene_blue"),
        os.path.join(base_dir, "data", "raw", "congo_red"),
        os.path.join(base_dir, "data", "raw", "crystal_violet"),
        os.path.join(base_dir, "data", "processed"),
        os.path.join(base_dir, "models", "baseline"),
        os.path.join(base_dir, "models", "cnn"),
        os.path.join(base_dir, "results", "plots"),
        os.path.join(base_dir, "results", "reports"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    logging.info(f"Ensured {len(dirs)} directories exist under '{base_dir}'")


# ── Dye Configuration ────────────────────────────────────────────────
# Each dye maps to its base RGB color at maximum concentration and
# the background color (white / near-white) at zero concentration.
# These follow simplified Beer-Lambert absorption relationships.

DYE_CONFIG = {
    "methylene_blue": {
        "name": "Methylene Blue",
        "rgb_max": (0, 50, 160),      # Deep blue at high ppm
        "rgb_min": (240, 245, 250),    # Near-white at 0 ppm
        "ppm_range": (0, 200),
    },
    "congo_red": {
        "name": "Congo Red",
        "rgb_max": (180, 20, 20),      # Deep red at high ppm
        "rgb_min": (255, 240, 240),    # Near-white at 0 ppm
        "ppm_range": (0, 200),
    },
    "crystal_violet": {
        "name": "Crystal Violet",
        "rgb_max": (80, 10, 130),      # Deep purple at high ppm
        "rgb_min": (245, 240, 255),    # Near-white at 0 ppm
        "ppm_range": (0, 200),
    },
}
