"""
PyTorch custom Dataset module for the Semiconductor Image Restoration System.
Implements the SemiconductorDataset class with support for normalized float tensors
and synchronized data augmentations for aligned image pairs.
"""

import logging
import random
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from dataset_manager.config import AugmentationConfig
import albumentations as A

logger = logging.getLogger(__name__)

class SemiconductorDataset(Dataset):
    """PyTorch Dataset class for handling paired clean (ground truth) and noisy (input) semiconductor images."""

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        base_dir: Path,
        augmentation_opts: Optional[AugmentationConfig] = None,
        is_training: bool = False,
    ):
        """Initializes the dataset.

        Args:
            metadata_df (pd.DataFrame): Dataframe containing dataset records.
            base_dir (Path): Base directory path to resolve paths in metadata database.
            augmentation_opts (AugmentationConfig, optional): Sync augmentations settings.
            is_training (bool): If True, enables random data augmentations.
        """
        self.df = metadata_df.reset_index(drop=True)
        self.base_dir = Path(base_dir)
        self.aug = augmentation_opts
        self.is_training = is_training
        
        # Build Albumentations composition pipeline
        self.transform = None
        if self.aug and self.aug.enabled and self.is_training:
            transforms_list = []
            if self.aug.horizontal_flip:
                transforms_list.append(A.HorizontalFlip(p=0.5))
            if self.aug.vertical_flip:
                transforms_list.append(A.VerticalFlip(p=0.5))
            if self.aug.random_rotation:
                # RandomRotate90 guarantees 90, 180, 270 degree rotatios, preserving pixel alignments without subpixel interpolation
                transforms_list.append(A.RandomRotate90(p=0.5))
            if self.aug.random_crop:
                th, tw = self.aug.crop_size
                transforms_list.append(A.RandomCrop(height=th, width=tw, p=1.0))
                
            self.transform = A.Compose(
                transforms_list,
                additional_targets={'image_gt': 'image'}
            )
            
        logger.info(
            f"Initialized SemiconductorDataset with {len(self.df)} samples. "
            f"Augmentations enabled: {self.transform is not None}."
        )

    def __len__(self) -> int:
        """Returns total number of items in this dataset.

        Returns:
            int: The dataset length.
        """
        return len(self.df)

    def _sync_augment(self, inp_img: np.ndarray, gt_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Applies identical, synchronized augmentations to both input and ground truth images.

        Uses the Albumentations composition created during initialization.

        Args:
            inp_img (np.ndarray): Noisy input.
            gt_img (np.ndarray): Clean ground truth target.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Augmented (input, ground_truth) images.
        """
        if not self.transform:
            return inp_img, gt_img

        augmented = self.transform(image=inp_img, image_gt=gt_img)
        return augmented["image"], augmented["image_gt"]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Loads and processes the item matching the index.

        Args:
            idx (int): The index.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Input tensor of shape (1, H, W) normalized to [0.0, 1.0].
                - Ground truth tensor of shape (1, H, W) normalized to [0.0, 1.0].
        """
        row = self.df.iloc[idx]
        
    def _load_array(self, path_str: str, frame_index: int = -1) -> np.ndarray:
        """Loads an image from a standard filesystem path or an HDF5 virtual path."""
        import h5py
        
        if "::" in path_str:
            file_path, internal_key = path_str.split("::", 1)
            fpath = Path(file_path)
            if not fpath.is_absolute():
                fpath = self.base_dir / fpath
                
            if not fpath.exists():
                raise FileNotFoundError(f"HDF5 file does not exist: {fpath}")
                
            try:
                with h5py.File(fpath, 'r') as f:
                    # Handle keys robustly
                    key_to_try = internal_key
                    if key_to_try not in f:
                        alt_key = f"/{internal_key.lstrip('/')}"
                        if alt_key in f:
                            key_to_try = alt_key
                        else:
                            raise KeyError(f"Internal key '{internal_key}' not found in {fpath}")
                            
                    dataset = f[key_to_try]
                    if not isinstance(dataset, h5py.Dataset):
                        raise ValueError(f"Object at '{internal_key}' is not an HDF5 dataset in {fpath}")
                        
                    if frame_index != -1 and len(dataset.shape) == 3 and dataset.shape[-1] not in [1, 3]:
                        img = dataset[frame_index]
                    else:
                        img = dataset[()]
            except Exception as e:
                raise RuntimeError(f"Failed to read HDF5 {fpath}::'{internal_key}': {e}")
        else:
            fpath = Path(path_str)
            if not fpath.is_absolute():
                fpath = self.base_dir / fpath
                
            if not fpath.exists():
                raise FileNotFoundError(f"Image file does not exist: {fpath}")
                
            img = cv2.imread(str(fpath), cv2.IMREAD_ANYDEPTH)
            if img is None:
                raise ValueError(f"Failed to load standard image: {fpath}")

        # Standardize to 2D (H, W)
        if len(img.shape) == 3:
            if img.shape[0] in [1, 3]:
                img = np.transpose(img, (1, 2, 0))
            if img.shape[-1] == 1:
                img = np.squeeze(img, axis=-1)
            elif img.shape[-1] == 3:
                # Convert RGB to Grayscale
                if img.dtype == np.uint16:
                    # cvtColor struggles with uint16 sometimes, handle manually if needed, but typically IMREAD_ANYDEPTH works
                    img = (img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114).astype(np.uint16)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        elif len(img.shape) != 2:
            raise ValueError(f"Unsupported array shape {img.shape} from {path_str}")

        # Support existing 128x128 resize logic for transformers
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
        return img

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Loads and processes the item matching the index.

        Args:
            idx (int): The index.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Input tensor of shape (1, H, W) normalized to [0.0, 1.0].
                - Ground truth tensor of shape (1, H, W) normalized to [0.0, 1.0].
        """
        row = self.df.iloc[idx]
        
        inp_str = str(row["input_path"])
        gt_str = str(row["ground_truth_path"])
        frame_idx = int(row.get("frame_index", -1))

        try:
            inp_img = self._load_array(inp_str, frame_idx)
            gt_img = self._load_array(gt_str, frame_idx)
        except Exception as e:
            raise ValueError(f"Failed to load image files at index {idx}: {e}")

        # Apply Synced Augmentation
        inp_img, gt_img = self._sync_augment(inp_img, gt_img)

        # Standardize float normalizations: convert [0, 255] uint8 -> [0.0, 1.0] float32
        # Preserve bit depth until tensor conversion
        inp_max = 65535.0 if inp_img.dtype == np.uint16 else 255.0
        gt_max = 65535.0 if gt_img.dtype == np.uint16 else 255.0
        inp_tensor = torch.from_numpy(inp_img.astype(np.float32)).float() / inp_max
        gt_tensor = torch.from_numpy(gt_img.astype(np.float32)).float() / gt_max

        # Expand channel dimension: H, W -> 1, H, W
        inp_tensor = inp_tensor.unsqueeze(0)
        gt_tensor = gt_tensor.unsqueeze(0)

        return inp_tensor, gt_tensor

    def get_sample_metadata(self, idx: int) -> Dict[str, Any]:
        """Retrieves raw metadata matching this index for diagnostic logs.

        Args:
            idx (int): The index.

        Returns:
            Dict[str, Any]: The pandas row as dictionary.
        """
        return self.df.iloc[idx].to_dict()
