"""
Preprocessor module for the Semiconductor Image Dataset System.
Features:
1. Dataset type auto-detection.
2. Aspect-ratio preserving resizing (letterboxing / cropping / stretching).
3. Automated pairing of noisy inputs and clean ground truths.
4. Splitting into train, validation, and test datasets.
"""

import logging
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import cv2
import numpy as np
from tqdm import tqdm

from dataset_manager.config import PipelineConfig
from dataset_manager.converter import ImageConverter
from dataset_manager.registry import DatasetRegistry

logger = logging.getLogger(__name__)

class DatasetPreprocessor:
    """Orchestrates dataset detection, format conversion, resizing, pairing, and folder splits."""

    VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}

    def __init__(self, config: PipelineConfig):
        """Initializes preprocessor with loaded configurations.

        Args:
            config (PipelineConfig): Global configuration instance.
        """
        self.config = config
        self.raw_dir = Path(config.paths.raw_dir)
        self.processed_dir = Path(config.paths.processed_dir)
        
        # Enforce output directories
        self.splits = ["train", "val", "test"]
        self.subfolders = ["input", "ground_truth"]

    def _get_dataset_config_for_path(self, dataset_path: Path) -> Optional[Any]:
        """Looks up matching DatasetSourceConfig based on path or name."""
        abs_path = dataset_path.resolve()
        for ds_key, ds_cfg in self.config.datasets.items():
            cfg_path = Path(ds_cfg.local_path).resolve()
            if cfg_path == abs_path or cfg_path.name.upper() == dataset_path.name.upper():
                return ds_cfg
        return None

    @staticmethod
    def detect_dataset_type(dataset_path: Path) -> str:
        """Automatically detects dataset type based on directory name and registry configurations.

        Args:
            dataset_path (Path): Path to the dataset subdirectory.

        Returns:
            str: Resolved Dataset Type name.
        """
        name, _ = DatasetRegistry.detect_and_resolve(dataset_path)
        return name

    @staticmethod
    def resize_preserving_aspect_ratio(
        image: np.ndarray,
        target_w: int,
        target_h: int,
        mode: str = "pad",
        interpolation_str: str = "lanczos",
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resizes grayscale image while preserving its aspect ratio.

        Calculates calibration offsets. Helpful for semiconductor feature metrology.

        Args:
            image (np.ndarray): Grayscale image 2D array.
            target_w (int): Target width.
            target_h (int): Target height.
            mode (str): 'pad' (letterbox), 'crop' (centered crop), 'stretch' (standard resize).
            interpolation_str (str): Interpolation method ('lanczos', 'area', 'cubic', 'linear').

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]:
                - Preprocessed image (np.ndarray)
                - Spatial transformation dictionary containing scale factors, offsets, and original size.
        """
        h_orig, w_orig = image.shape[:2]
        
        # Select interpolation
        interp_map = {
            "lanczos": cv2.INTER_LANCZOS4,
            "area": cv2.INTER_AREA,
            "cubic": cv2.INTER_CUBIC,
            "linear": cv2.INTER_LINEAR,
        }
        interp = interp_map.get(interpolation_str.lower(), cv2.INTER_LANCZOS4)

        transform_meta = {
            "original_height": h_orig,
            "original_width": w_orig,
            "mode": mode,
            "scale_x": 1.0,
            "scale_y": 1.0,
            "offset_x": 0,
            "offset_y": 0,
        }

        if mode == "stretch":
            resized = cv2.resize(image, (target_w, target_h), interpolation=interp)
            transform_meta["scale_x"] = target_w / w_orig
            transform_meta["scale_y"] = target_h / h_orig
            return resized, transform_meta

        elif mode == "crop":
            # Scale short side to fill target dim
            scale = max(target_w / w_orig, target_h / h_orig)
            w_scaled = int(w_orig * scale)
            h_scaled = int(h_orig * scale)
            
            resized_full = cv2.resize(image, (w_scaled, h_scaled), interpolation=interp)
            
            # Extract center crop
            offset_x = (w_scaled - target_w) // 2
            offset_y = (h_scaled - target_h) // 2
            
            cropped = resized_full[offset_y : offset_y + target_h, offset_x : offset_x + target_w]
            
            transform_meta["scale_x"] = scale
            transform_meta["scale_y"] = scale
            transform_meta["offset_x"] = -offset_x
            transform_meta["offset_y"] = -offset_y
            return cropped, transform_meta

        else:  # mode == 'pad' (letterbox)
            # Scale long side to fit within target dimensions
            scale = min(target_w / w_orig, target_h / h_orig)
            w_scaled = int(w_orig * scale)
            h_scaled = int(h_orig * scale)
            
            resized_sub = cv2.resize(image, (w_scaled, h_scaled), interpolation=interp)
            
            # Create a blank black canvas
            canvas = np.zeros((target_h, target_w), dtype=np.uint8)
            
            # Locate centered coordinates
            offset_x = (target_w - w_scaled) // 2
            offset_y = (target_h - h_scaled) // 2
            
            canvas[offset_y : offset_y + h_scaled, offset_x : offset_x + w_scaled] = resized_sub
            
            transform_meta["scale_x"] = scale
            transform_meta["scale_y"] = scale
            transform_meta["offset_x"] = offset_x
            transform_meta["offset_y"] = offset_y
            return canvas, transform_meta

    def pair_images(self, dataset_type: str, dataset_path: Path) -> List[Tuple[Path, Path]]:
        """Scans a raw dataset directory and automatically pairs noisy files with clean targets
        by delegating to the appropriate handler registered in the DatasetRegistry.

        Args:
            dataset_type (str): The dataset identifier (SEM, NIST, SIDD, SBF_SEM, Synthetic).
            dataset_path (Path): Path to the raw directory.

        Returns:
            List[Tuple[Path, Path]]: A list of (noisy_input_path, ground_truth_path).
        """
        handler = DatasetRegistry.get_handler(dataset_type)
        pairs = handler.pair_images(dataset_path)
        logger.info(f"Registry-resolved pairing matched {len(pairs)} image pairs for type {dataset_type}.")
        return pairs

    def split_dataset(
        self, pairs: List[Tuple[Path, Path]]
    ) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
        """Splits the image pairs into train, Validation, and test datasets.

        Uses the seed in configuration to keep partition selections reproducible.

        Args:
            pairs (List[Tuple[Path, Path]]): Full paired dataset list.

        Returns:
            Tuple[List, List, List]: Split partitions: (train_pairs, val_pairs, test_pairs).
        """
        # Set seed
        random.seed(self.config.preprocessing.split.seed)
        
        # Shallow copy to protect input list order
        shuffled = list(pairs)
        random.shuffle(shuffled)

        total = len(shuffled)
        r_train = self.config.preprocessing.split.train
        r_val = self.config.preprocessing.split.val
        
        idx_train = int(total * r_train)
        idx_val = idx_train + int(total * r_val)

        train_pairs = shuffled[:idx_train]
        val_pairs = shuffled[idx_train:idx_val]
        test_pairs = shuffled[idx_val:]

        logger.info(
            f"Dataset split: Train = {len(train_pairs)}, Val = {len(val_pairs)}, Test = {len(test_pairs)}"
        )
        return train_pairs, val_pairs, test_pairs

    def process_and_save_pair(
        self,
        pair: Tuple[Path, Path],
        split: str,
        dataset_name: str,
        pair_index: int,
        version: str = "1.0",
    ) -> Optional[Dict[str, Any]]:
        """Loads, converts to grayscale, resizes, and saves a single noisy/clean pair.

        Args:
            pair (Tuple[Path, Path]): (noisy_input_path, ground_truth_path).
            split (str): 'train', 'val', or 'test'.
            dataset_name (str): Label identifier.
            pair_index (int): A unique counter for renaming files.

        Returns:
            Optional[Dict[str, Any]]: Metadata record dictionary.
        """
        input_path, gt_path = pair
        
        try:
            # 1. Load and convert to grayscale values
            input_gray = ImageConverter.load_as_grayscale(input_path)
            gt_gray = ImageConverter.load_as_grayscale(gt_path)
            
            # Validate shapes match
            if input_gray.shape != gt_gray.shape:
                logger.warning(
                    f"Dimension mismatch between input {input_path.name} {input_gray.shape} "
                    f"and ground truth {gt_path.name} {gt_gray.shape}. Resizing ground truth to fit input before pipeline resize."
                )
                h_in, w_in = input_gray.shape[:2]
                gt_gray = cv2.resize(gt_gray, (w_in, h_in), interpolation=cv2.INTER_AREA)

            # 2. Apply Aspect-Ratio Preserving Resizer
            resize_opts = self.config.preprocessing.resize
            if resize_opts.enabled:
                input_processed, input_meta = self.resize_preserving_aspect_ratio(
                    input_gray,
                    resize_opts.width,
                    resize_opts.height,
                    mode=resize_opts.mode,
                    interpolation_str=resize_opts.interpolation,
                )
                gt_processed, gt_meta = self.resize_preserving_aspect_ratio(
                    gt_gray,
                    resize_opts.width,
                    resize_opts.height,
                    mode=resize_opts.mode,
                    interpolation_str=resize_opts.interpolation,
                )
            else:
                input_processed = input_gray
                gt_processed = gt_gray
                input_meta = {"original_height": input_gray.shape[0], "original_width": input_gray.shape[1]}
                gt_meta = {"original_height": gt_gray.shape[0], "original_width": gt_gray.shape[1]}

            # 3. Create destination subfolders
            dest_input_dir = self.processed_dir / split / "input"
            dest_gt_dir = self.processed_dir / split / "ground_truth"
            dest_input_dir.mkdir(parents=True, exist_ok=True)
            dest_gt_dir.mkdir(parents=True, exist_ok=True)

            # Define output names to prevent overlaps across datasets
            extension = ".png" # Forces lossless standardization saving
            out_filename = f"{dataset_name}_{split}_{pair_index:05d}{extension}"
            
            out_input_path = dest_input_dir / out_filename
            out_gt_path = dest_gt_dir / out_filename

            # 4. Save PNG files
            cv2.imwrite(str(out_input_path), input_processed)
            cv2.imwrite(str(out_gt_path), gt_processed)

            # Get final dimensions
            final_h, final_w = input_processed.shape[:2]

            # Return CSV metadata structure
            return {
                "image_name": out_filename,
                "dataset_name": dataset_name,
                "version": version,
                "width": final_w,
                "height": final_h,
                "image_type": input_path.suffix.lstrip(".").upper(),
                "ground_truth_path": str(out_gt_path.relative_to(self.processed_dir.parent.parent)),
                "input_path": str(out_input_path.relative_to(self.processed_dir.parent.parent)),
                "split": split,
                "original_width": input_meta["original_width"],
                "original_height": input_meta["original_height"],
                "scale_factor_x": input_meta.get("scale_x", 1.0),
                "scale_factor_y": input_meta.get("scale_y", 1.0),
                "offset_x": input_meta.get("offset_x", 0),
                "offset_y": input_meta.get("offset_y", 0),
                "noise_type": "",
                "noise_level": 0.0,
                "blur_level": 0.0,
                "resolution_loss": 0.0,
                "entropy": 0.0,
                "texture_score": 0.0,
                "edge_density": 0.0,
                "fingerprint": "",
            }

        except Exception as e:
            logger.error(f"Failed to process pair ({input_path.name}, {gt_path.name}): {e}")
            return None

    def execute(self) -> List[Dict[str, Any]]:
        """Main orchestrator that runs the entire preprocessing pipeline.

        1. Walks raw folders.
        2. Detects dataset type.
        3. If SBF-SEM TIFF stacks are found, slices them.
        4. Pairs images.
        5. Splits data into train/val/test splits.
        6. Processes, standardizes, and writes images to the 'processed' directory.
        7. Returns metadata records.

        Returns:
            List[Dict[str, Any]]: Preprocessed metadata listings.
        """
        logger.info("Initializing global Dataset Preprocessing Pipeline...")
        
        # Clean previous processed folder to ensure clean, isolated builds
        if self.processed_dir.exists():
            logger.warning(f"Output directory {self.processed_dir} already exists. Preparing clean rebuild...")
            shutil.rmtree(self.processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        if not self.raw_dir.exists():
            logger.error(f"Raw source directory {self.raw_dir} is missing.")
            return []

        # Find all primary folders under datasets
        dataset_paths = [d for d in self.raw_dir.iterdir() if d.is_dir() and d.name != "processed"]
        
        if not dataset_paths:
            logger.info("No raw datasets detected under datasets/ folder.")
            return []

        all_metadata_records: List[Dict[str, Any]] = []

        for dp in dataset_paths:
            # 1. Resolve dataset config, name and version
            ds_cfg = self._get_dataset_config_for_path(dp)
            if ds_cfg is not None:
                ds_name = ds_cfg.name
                ds_version = ds_cfg.version
                # detect the type for selecting appropriate Handler (e.g. SBF_SEM, SEM, NIST)
                ds_type = self.detect_dataset_type(dp)
            else:
                ds_type = self.detect_dataset_type(dp)
                ds_name = ds_type
                ds_version = "1.0"
                
            logger.info(f"Detected subfolder '{dp.name}' as dataset: {ds_name} (type: {ds_type}, version: {ds_version})")

            # 2. Trigger registry-resolved handler resource unpacking hooks (e.g. slicing TIFF stacks)
            handler = DatasetRegistry.get_handler(ds_type)
            handler.unpack_resources(dp)

            # 3. Pair clean and noisy files
            pairs = self.pair_images(ds_type, dp)
            if not pairs:
                logger.warning(f"No valid image pairs found for dataset {ds_type} in {dp}.")
                continue

            # 4. Split dataset
            train_p, val_p, test_p = self.split_dataset(pairs)

            # 5. Process and save pairs
            pair_counter = 0
            for split_name, split_pairs in [("train", train_p), ("val", val_p), ("test", test_p)]:
                desc = f"Processing {ds_name} -> Split: {split_name}"
                for p in tqdm(split_pairs, desc=desc):
                    meta = self.process_and_save_pair(p, split_name, ds_name, pair_counter, version=ds_version)
                    if meta:
                        all_metadata_records.append(meta)
                        pair_counter += 1

        logger.info(f"Pipeline finished! Preprocessed {len(all_metadata_records)} pairs.")
        return all_metadata_records
