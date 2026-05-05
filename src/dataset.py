"""
PyTorch Dataset & DataLoader
==============================
Custom Dataset class for loading dye sample images with ppm labels.
Handles train/val/test splitting and torchvision transforms.
"""

import os
import logging

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split


class DyeDataset(Dataset):
    """
    PyTorch Dataset for dye concentration images.
    
    Loads images and ppm labels from a metadata CSV file.
    Applies torchvision transforms for CNN input preparation.
    """
    
    def __init__(
        self,
        csv_path: str,
        root_dir: str = ".",
        transform=None,
        dye_type: str = None,
    ):
        """
        Args:
            csv_path: Path to metadata.csv
            root_dir: Root directory (image paths in CSV are relative to this)
            transform: torchvision transform pipeline
            dye_type: If specified, filter to only this dye type
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Load metadata
        self.df = pd.read_csv(csv_path)
        
        # Optional dye type filter
        if dye_type is not None:
            self.df = self.df[self.df["dye_type"] == dye_type].reset_index(drop=True)
        
        logging.info(f"Dataset loaded: {len(self.df)} samples"
                      f"{f' (filtered: {dye_type})' if dye_type else ''}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img_path = os.path.join(self.root_dir, row["image_path"])
        image = Image.open(img_path).convert("RGB")
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default: convert to tensor and normalize
            image = transforms.ToTensor()(image)
        
        # Target: ppm value as float tensor
        ppm = torch.tensor(row["ppm"], dtype=torch.float32)
        
        return image, ppm
    
    def get_ppm_stats(self):
        """Get mean and std of ppm values for target normalization."""
        return self.df["ppm"].mean(), self.df["ppm"].std()


def get_transforms(mode: str = "train", image_size: int = 224):
    """
    Get torchvision transform pipeline.
    
    Args:
        mode: 'train' (with augmentation) or 'val'/'test' (no augmentation)
        image_size: Target image size
    
    Returns:
        torchvision.transforms.Compose
    """
    # ImageNet normalization — REQUIRED for pretrained ResNet18 backbone
    imagenet_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    
    if mode == "train":
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),  # Handle angled smartphone shots
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05), # Strong lighting variance
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)), # Simulate out-of-focus camera
            transforms.ToTensor(),
            imagenet_normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            imagenet_normalize,
        ])


def get_data_loaders(
    csv_path: str,
    root_dir: str = ".",
    batch_size: int = 32,
    val_split: float = 0.15,
    test_split: float = 0.15,
    image_size: int = 224,
    seed: int = 42,
    dye_type: str = None,
):
    """
    Create train/val/test DataLoaders with proper splitting.
    
    Args:
        csv_path: Path to metadata CSV
        root_dir: Project root directory
        batch_size: Batch size for DataLoader
        val_split: Fraction for validation set
        test_split: Fraction for test set
        image_size: Image resize dimension
        seed: Random seed for reproducible splits
        dye_type: Optional dye type filter
    
    Returns:
        (train_loader, val_loader, test_loader, dataset)
    """
    # Create dataset with validation transforms (no augmentation for splitting)
    full_dataset = DyeDataset(
        csv_path=csv_path,
        root_dir=root_dir,
        transform=get_transforms("val", image_size),
        dye_type=dye_type,
    )
    
    # Create index splits
    indices = list(range(len(full_dataset)))
    
    # First split: train+val vs test
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_split,
        random_state=seed,
    )
    
    # Second split: train vs val
    relative_val_size = val_split / (1 - test_split)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=relative_val_size,
        random_state=seed,
    )
    
    logging.info(f"Data split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    
    # Create datasets with appropriate transforms
    train_dataset = DyeDataset(
        csv_path=csv_path,
        root_dir=root_dir,
        transform=get_transforms("train", image_size),
        dye_type=dye_type,
    )
    
    # Create subsets
    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(full_dataset, val_idx)
    test_subset = Subset(full_dataset, test_idx)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,   # Windows-safe
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    
    return train_loader, val_loader, test_loader, full_dataset
