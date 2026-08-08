"""
Custom PyTorch Dataset module for the Physics-Guided Image Analyzer (Module 7).
Parses metadata.csv and resolves detailed degradation JSON files.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Class index mappings
NOISE_TYPE_MAP = {
    "none": 0,
    "poisson_gaussian": 1,
    "sensor": 2,
    "e-beam_shot_noise": 3
}

BLUR_TYPE_MAP = {
    "none": 0,
    "gaussian": 1,
    "defocus": 2,
    "motion": 3
}

SEVERITY_MAP = {
    "Easy": 0,
    "Medium": 1,
    "Hard": 2,
    "Extreme": 3
}

# Inverse maps for decoding
REV_NOISE_MAP = {v: k for k, v in NOISE_TYPE_MAP.items()}
REV_BLUR_MAP = {v: k for k, v in BLUR_TYPE_MAP.items()}
REV_SEVERITY_MAP = {v: k for k, v in SEVERITY_MAP.items()}


class ImageAnalyzerDataset(Dataset):
    """PyTorch Dataset loading semiconductor images and resolving their multi-task degradation labels."""

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        base_dir: Path,
        image_size: Tuple[int, int] = (128, 128)
    ) -> None:
        """Initializes the dataset.

        Args:
            metadata_df (pd.DataFrame): Input metadata records.
            base_dir (Path): Base directory path to resolve image file locations.
            image_size (Tuple[int, int]): Size to resize images to for CPU training.
        """
        self.df = metadata_df.reset_index(drop=True)
        self.base_dir = Path(base_dir)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        row = self.df.iloc[idx]
        
        # 1. Load the preprocessed input image
        inp_relative = Path(row["input_path"])
        inp_full = self.base_dir / inp_relative
        
        def _load_image(path_str: str):
            if ".h5::" in path_str or ".hdf5::" in path_str:
                import h5py
                file_path, group_path = path_str.split("::")
                with h5py.File(file_path, 'r') as f:
                    return f[group_path][()]
            else:
                return cv2.imread(path_str, cv2.IMREAD_ANYDEPTH)

        img = _load_image(str(inp_full))
        if img is None:
            raise ValueError(f"Failed to read image at: {inp_full}")

        # Load the preprocessed ground truth image
        gt_relative = Path(row["ground_truth_path"])
        gt_full = self.base_dir / gt_relative
        gt_img = _load_image(str(gt_full))
        if gt_img is None:
            raise ValueError(f"Failed to read ground truth image at: {gt_full}")

        # Compute physical image properties: Brightness and Contrast
        inp_max = 65535.0 if img.dtype == np.uint16 else 255.0
        gt_max = 65535.0 if gt_img.dtype == np.uint16 else 255.0
        
        brightness_val = float(np.mean(img)) / inp_max
        contrast_val = float(np.std(img)) / inp_max

        # Resize images for fast CPU processing
        img_resized = cv2.resize(img.astype(np.float32), self.image_size, interpolation=cv2.INTER_AREA)
        gt_resized = cv2.resize(gt_img.astype(np.float32), self.image_size, interpolation=cv2.INTER_AREA)
        
        # Convert to tensors: shape [1, H, W] in range [0.0, 1.0]
        img_tensor = torch.from_numpy(img_resized).float().unsqueeze(0) / inp_max
        gt_tensor = torch.from_numpy(gt_resized).float().unsqueeze(0) / gt_max

        # 2. Resolve target labels
        noise_type_str = str(row.get("noise_type", "none")).lower()
        noise_level_val = float(row.get("noise_level", 0.0))
        blur_strength_val = float(row.get("blur_level", 0.0))
        resolution_loss_val = float(row.get("resolution_loss", 0.0))
        edge_density_val = float(row.get("edge_density", 0.0))
        texture_val = float(row.get("texture_score", 0.0))
        entropy_val = float(row.get("entropy", 0.0))

        # Default values for metadata JSON attributes
        blur_type_str = "none"
        severity_str = "Easy"

        # Attempt to load synthetic degradation JSON metadata for precise blur type and severity level
        # File name convention: Synthetic_Generated_train_00001.png -> synthetic_00001_meta.json
        image_name = str(row["image_name"])
        if "Synthetic_Generated" in image_name:
            try:
                # Extract index
                parts = image_name.split("_")
                idx_str = parts[-1].split(".")[0] # e.g. '00001'
                
                # Check under raw datasets dir
                meta_json_path = self.base_dir / "datasets" / "Synthetic_Generated" / f"synthetic_{idx_str}_meta.json"
                if meta_json_path.exists():
                    with open(meta_json_path, "r", encoding="utf-8") as f:
                        meta_data = json.load(f)
                    
                    severity_str = meta_data.get("severity_level", "Easy")
                    
                    # Extract blur type
                    params = meta_data.get("parameters", {})
                    for key, val in params.items():
                        if "blur" in key:
                            blur_type_str = val.get("blur_type", "none")
                            break
            except Exception as e:
                logger.warning(f"Error parsing metadata JSON for sample {image_name}: {e}")

        # Map strings to indices
        noise_type_idx = NOISE_TYPE_MAP.get(noise_type_str, 0)
        blur_type_idx = BLUR_TYPE_MAP.get(blur_type_str, 0)
        severity_idx = SEVERITY_MAP.get(severity_str, 0)

        # Scale continuous values to safe ranges
        # Normalize texture score (typically ~0 to 60000 in metadata)
        texture_norm = min(1.0, texture_val / 60000.0)
        # Normalize entropy (typically 0 to 8.0)
        entropy_norm = min(1.0, entropy_val / 8.0)

        # Confidence target: standard 1.0 for ground truth synthetic dataset
        confidence_val = 1.0

        targets = {
            "noise_type": torch.tensor(noise_type_idx, dtype=torch.long),
            "blur_type": torch.tensor(blur_type_idx, dtype=torch.long),
            "severity": torch.tensor(severity_idx, dtype=torch.long),
            "noise_level": torch.tensor(noise_level_val, dtype=torch.float32),
            "blur_strength": torch.tensor(blur_strength_val, dtype=torch.float32),
            "resolution_loss": torch.tensor(resolution_loss_val, dtype=torch.float32),
            "compression_quality": torch.tensor(1.0, dtype=torch.float32), # Defaults to 1.0 (uncompressed)
            "brightness": torch.tensor(brightness_val, dtype=torch.float32),
            "contrast": torch.tensor(contrast_val, dtype=torch.float32),
            "gamma": torch.tensor(1.0, dtype=torch.float32), # Defaults to 1.0
            "edge_density": torch.tensor(edge_density_val, dtype=torch.float32),
            "texture_complexity": torch.tensor(texture_norm, dtype=torch.float32),
            "entropy": torch.tensor(entropy_norm, dtype=torch.float32),
            "confidence": torch.tensor(confidence_val, dtype=torch.float32)
        }

        return img_tensor, gt_tensor, targets
