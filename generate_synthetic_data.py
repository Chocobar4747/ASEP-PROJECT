"""
Synthetic Dye Sample Image Generator
=====================================
Generates realistic synthetic images simulating dyed fabric/solution samples
at various concentrations (ppm). Uses Beer-Lambert law inspired color mapping.

Each image is a 224x224 patch with:
- Base color interpolated between white (0 ppm) and dye color (max ppm)
- Gaussian noise to simulate camera sensor noise
- Slight brightness variation to simulate lighting differences
- Circular mask to simulate a petri dish / cuvette view
"""

import os
import sys
import logging
import argparse

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.utils import set_seed, setup_logging, ensure_dirs, DYE_CONFIG


def concentration_to_rgb(ppm: float, dye_key: str) -> tuple:
    """
    Map concentration (ppm) to an RGB color using Beer-Lambert-like
    exponential absorption curve.
    
    At 0 ppm → near-white (rgb_min)
    At max ppm → fully saturated dye color (rgb_max)
    The transition follows: color = min + (max - min) * (1 - exp(-k * ppm))
    """
    cfg = DYE_CONFIG[dye_key]
    ppm_min, ppm_max = cfg["ppm_range"]
    rgb_min = np.array(cfg["rgb_min"], dtype=np.float64)
    rgb_max = np.array(cfg["rgb_max"], dtype=np.float64)
    
    # Normalize ppm to [0, 1]
    t = np.clip((ppm - ppm_min) / (ppm_max - ppm_min), 0, 1)
    
    # Exponential absorption curve (Beer-Lambert inspired)
    k = 3.0  # absorption coefficient — controls curve steepness
    absorption = 1.0 - np.exp(-k * t)
    
    # Interpolate color
    rgb = rgb_min + (rgb_max - rgb_min) * absorption
    return tuple(np.clip(rgb, 0, 255).astype(int))


def generate_sample_image(
    ppm: float,
    dye_key: str,
    size: int = 224,
    noise_std: float = 5.0,
    brightness_var: float = 10.0,
) -> Image.Image:
    """
    Generate a single synthetic dyed-sample image.
    
    Args:
        ppm: Concentration in parts per million
        dye_key: Key from DYE_CONFIG
        size: Image dimension (square)
        noise_std: Standard deviation of Gaussian sensor noise
        brightness_var: Max brightness offset to simulate lighting variation
    
    Returns:
        PIL Image (RGB, size x size)
    """
    base_rgb = concentration_to_rgb(ppm, dye_key)
    
    # Create base image array
    img_array = np.full((size, size, 3), base_rgb, dtype=np.float64)
    
    # Add Gaussian noise (camera sensor simulation)
    noise = np.random.normal(0, noise_std, img_array.shape)
    img_array += noise
    
    # Add slight brightness variation (lighting simulation)
    brightness_offset = np.random.uniform(-brightness_var, brightness_var)
    img_array += brightness_offset
    
    # Add subtle radial gradient (vignette effect — simulates lens)
    y, x = np.ogrid[:size, :size]
    center = size / 2
    dist = np.sqrt((x - center) ** 2 + (y - center) ** 2)
    max_dist = np.sqrt(2) * center
    vignette = 1.0 - 0.15 * (dist / max_dist) ** 2
    img_array *= vignette[:, :, np.newaxis]
    
    # Clip and convert to uint8
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_array, "RGB")
    
    # Draw circular mask (petri dish simulation)
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    margin = 8
    draw.ellipse([margin, margin, size - margin, size - margin], fill=255)
    
    # Create white background
    background = Image.new("RGB", (size, size), (245, 245, 245))
    background.paste(img, mask=mask)
    
    # Slight blur at edges for realism
    background = background.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    return background


def generate_dataset(
    output_dir: str = ".",
    samples_per_dye: int = 500,
    seed: int = 42,
):
    """
    Generate full synthetic dataset for all configured dyes.
    
    Creates images in data/raw/<dye_type>/ and a metadata CSV at data/metadata.csv.
    """
    set_seed(seed)
    ensure_dirs(output_dir)
    
    records = []
    
    for dye_key, cfg in DYE_CONFIG.items():
        dye_name = cfg["name"]
        ppm_min, ppm_max = cfg["ppm_range"]
        
        logging.info(f"Generating {samples_per_dye} samples for {dye_name} "
                      f"({ppm_min}–{ppm_max} ppm)...")
        
        # Generate evenly spaced ppm values with small random jitter
        base_ppms = np.linspace(ppm_min, ppm_max, samples_per_dye)
        jitter = np.random.uniform(-0.5, 0.5, samples_per_dye)
        ppms = np.clip(base_ppms + jitter, ppm_min, ppm_max)
        
        dye_dir = os.path.join(output_dir, "data", "raw", dye_key)
        os.makedirs(dye_dir, exist_ok=True)
        
        for i, ppm in enumerate(ppms):
            # Generate image
            img = generate_sample_image(ppm, dye_key)
            
            # Save
            filename = f"{dye_key}_{i:04d}.png"
            filepath = os.path.join(dye_dir, filename)
            img.save(filepath)
            
            # Extract mean RGB for metadata
            img_array = np.array(img).astype(np.float64)
            r_mean = img_array[:, :, 0].mean()
            g_mean = img_array[:, :, 1].mean()
            b_mean = img_array[:, :, 2].mean()
            
            records.append({
                "image_path": os.path.join("data", "raw", dye_key, filename),
                "dye_type": dye_key,
                "dye_name": dye_name,
                "ppm": round(ppm, 2),
                "r_mean": round(r_mean, 2),
                "g_mean": round(g_mean, 2),
                "b_mean": round(b_mean, 2),
            })
            
            if (i + 1) % 100 == 0:
                logging.info(f"  [{dye_name}] Generated {i + 1}/{samples_per_dye}")
    
    # Save metadata CSV
    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "data", "metadata.csv")
    df.to_csv(csv_path, index=False)
    
    logging.info(f"\n{'='*60}")
    logging.info(f"Dataset generation complete!")
    logging.info(f"  Total images: {len(df)}")
    logging.info(f"  Dye types: {df['dye_type'].nunique()}")
    logging.info(f"  PPM range: {df['ppm'].min():.1f} – {df['ppm'].max():.1f}")
    logging.info(f"  Metadata saved to: {csv_path}")
    logging.info(f"{'='*60}")
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic dye sample dataset")
    parser.add_argument("--output-dir", type=str, default=".", help="Project root directory")
    parser.add_argument("--samples", type=int, default=500, help="Samples per dye type")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    setup_logging()
    generate_dataset(
        output_dir=args.output_dir,
        samples_per_dye=args.samples,
        seed=args.seed,
    )
