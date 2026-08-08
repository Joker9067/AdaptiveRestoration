"""
DataLoader factory module for the Semiconductor Image Restoration System.
Constructs optimized PyTorch DataLoaders for train, validation, and test splits.
"""

import logging
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
from torch.utils.data import DataLoader

from dataset_manager.config import PipelineConfig
from dataset_manager.dataset import SemiconductorDataset
from dataset_manager.metadata import MetadataManager

logger = logging.getLogger(__name__)

def create_dataloaders(
    metadata_csv_path: Path,
    config: PipelineConfig,
    base_dir: Optional[Path] = None,
) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    """Loads metadata CSV and instantiates DataLoaders for Train, Validation, and Test splits.

    Args:
        metadata_csv_path (Path): Path to metadata.csv.
        config (PipelineConfig): Global configuration instance.
        base_dir (Path, optional): Directory to resolve relative paths.
                                  Defaults to the workspace root directory.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: (train_loader, val_loader, test_loader).
                                                  Returns None for splits that contain zero images.
    """
    if base_dir is None:
        # Defaults to the directory containing metadata file (workspace root)
        base_dir = metadata_csv_path.parent.parent.parent

    # 1. Initialize metadata reader and parse files
    meta_mgr = MetadataManager(metadata_csv_path)
    try:
        df = meta_mgr.load_metadata()
    except Exception as e:
        logger.error(f"Failed to load dataset registry metadata: {e}")
        return None, None, None

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    logger.info(
        f"Loading database records. Total samples: {len(df)}. "
        f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
    )

    # 2. Build datasets
    train_loader, val_loader, test_loader = None, None, None

    dl_opts = config.dataloader
    aug_opts = config.augmentation

    # Safety overrides for Python windows multiprocessing overheads
    # If using 0 or 1 worker, standard execution. If dataset size is tiny, avoid multiple worker overheads
    def get_safe_num_workers(split_len: int) -> int:
        if split_len < dl_opts.batch_size * 2:
            return 0
        return dl_opts.num_workers

    # Train loader setup
    if len(train_df) > 0:
        train_ds = SemiconductorDataset(
            metadata_df=train_df,
            base_dir=base_dir,
            augmentation_opts=aug_opts,
            is_training=True,
        )
        safe_workers = get_safe_num_workers(len(train_df))
        train_loader = DataLoader(
            train_ds,
            batch_size=min(dl_opts.batch_size, len(train_df)),
            shuffle=dl_opts.shuffle_train,
            num_workers=safe_workers,
            pin_memory=dl_opts.pin_memory,
            drop_last=dl_opts.drop_last if len(train_df) > dl_opts.batch_size else False,
        )

    # Validation loader setup
    if len(val_df) > 0:
        val_ds = SemiconductorDataset(
            metadata_df=val_df,
            base_dir=base_dir,
            augmentation_opts=None,
            is_training=False,
        )
        safe_workers = get_safe_num_workers(len(val_df))
        val_loader = DataLoader(
            val_ds,
            batch_size=min(dl_opts.batch_size, len(val_df)),
            shuffle=False,
            num_workers=safe_workers,
            pin_memory=dl_opts.pin_memory,
            drop_last=False,
        )

    # Test loader setup
    if len(test_df) > 0:
        test_ds = SemiconductorDataset(
            metadata_df=test_df,
            base_dir=base_dir,
            augmentation_opts=None,
            is_training=False,
        )
        safe_workers = get_safe_num_workers(len(test_df))
        test_loader = DataLoader(
            test_ds,
            batch_size=min(dl_opts.batch_size, len(test_df)),
            shuffle=False,
            num_workers=safe_workers,
            pin_memory=dl_opts.pin_memory,
            drop_last=False,
        )

    return train_loader, val_loader, test_loader
