"""
Dataset Statistics Generator module for the Semiconductor Image Restoration System.
Analyzes processed files to output total counts, dimension profiles,
intensity distributions, means, standard deviations, and grayscale value histograms,
separated for Noisy (inputs) and Clean (ground truths) across splits.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class DatasetStatsGenerator:
    """Computes numerical distributions, counts, formats, intensity means, and std metrics for clean/noisy datasets."""

    def __init__(self, metadata_path: Path):
        """Initializes with preprocessed metadata CSV file path.

        Args:
            metadata_path (Path): Path to metadata.csv.
        """
        self.metadata_path = Path(metadata_path)

    def generate_statistics(self) -> Dict[str, Any]:
        """Iterates preprocessed images to calculate detailed metrics.

        Returns:
            Dict[str, Any]: Compiled statistics dictionary.
        """
        import pandas as pd
        stats = {
            "total_images": 0,
            "formats": {},
            "dimensions": {
                "avg_width": 0.0,
                "avg_height": 0.0,
                "min_width": 99999,
                "max_width": 0,
                "min_height": 99999,
                "max_height": 0,
            },
            "noisy_stats": {
                "mean": 0.0,
                "std": 0.0,
                "intensity_histogram_256": [0] * 256,
            },
            "clean_stats": {
                "mean": 0.0,
                "std": 0.0,
                "intensity_histogram_256": [0] * 256,
            },
            "datasets_summary": {},
            "splits_summary": {},
        }

        if not self.metadata_path.exists():
            logger.error(f"Metadata file missing for stats calculation: {self.metadata_path}")
            return stats

        try:
            df = pd.read_csv(self.metadata_path)
        except Exception as e:
            logger.error(f"Failed to parse metadata CSV for stats: {e}")
            return stats

        total_pairs = len(df)
        if total_pairs == 0:
            logger.warning("Metadata CSV contains 0 records. Statistics cannot be calculated.")
            return stats

        stats["total_images"] = total_pairs * 2  # Input + Ground Truth
        base_dir = self.metadata_path.parent.parent

        widths = []
        heights = []
        formats = {}

        # Accumulators for clean and noisy
        noisy_sum = 0.0
        noisy_sq_sum = 0.0
        noisy_pixels = 0
        noisy_hist = np.zeros(256, dtype=np.int64)

        clean_sum = 0.0
        clean_sq_sum = 0.0
        clean_pixels = 0
        clean_hist = np.zeros(256, dtype=np.int64)

        datasets_counts = {}
        splits_data = {}

        for _, row in df.iterrows():
            ds_name = row.get("dataset_name", "UNKNOWN")
            split = row.get("split", "UNKNOWN")
            
            datasets_counts[ds_name] = datasets_counts.get(ds_name, 0) + 1
            if split not in splits_data:
                splits_data[split] = {
                    "pairs_count": 0,
                    "noisy_sum": 0.0, "noisy_sq_sum": 0.0, "noisy_pixels": 0,
                    "clean_sum": 0.0, "clean_sq_sum": 0.0, "clean_pixels": 0
                }
            splits_data[split]["pairs_count"] += 1

            inp_rel = row.get("input_path")
            gt_rel = row.get("ground_truth_path")
            if not inp_rel or not gt_rel:
                continue

            inp_path = base_dir.parent / Path(inp_rel)
            gt_path = base_dir.parent / Path(gt_rel)

            # Process Noisy Input
            if inp_path.exists():
                suffix = inp_path.suffix.lower().replace(".", "")
                formats[suffix] = formats.get(suffix, 0) + 1
                img = cv2.imread(str(inp_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    h, w = img.shape[:2]
                    widths.append(w)
                    heights.append(h)
                    img_norm = img.astype(np.float64) / 255.0
                    
                    noisy_sum += np.sum(img_norm)
                    noisy_sq_sum += np.sum(img_norm ** 2)
                    noisy_pixels += img_norm.size
                    
                    # Accumulate split-wise
                    splits_data[split]["noisy_sum"] += np.sum(img_norm)
                    splits_data[split]["noisy_sq_sum"] += np.sum(img_norm ** 2)
                    splits_data[split]["noisy_pixels"] += img_norm.size

                    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
                    noisy_hist += hist.flatten().astype(np.int64)

            # Process Clean Ground Truth
            if gt_path.exists():
                suffix = gt_path.suffix.lower().replace(".", "")
                formats[suffix] = formats.get(suffix, 0) + 1
                img = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img_norm = img.astype(np.float64) / 255.0
                    
                    clean_sum += np.sum(img_norm)
                    clean_sq_sum += np.sum(img_norm ** 2)
                    clean_pixels += img_norm.size
                    
                    # Accumulate split-wise
                    splits_data[split]["clean_sum"] += np.sum(img_norm)
                    splits_data[split]["clean_sq_sum"] += np.sum(img_norm ** 2)
                    splits_data[split]["clean_pixels"] += img_norm.size

                    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
                    clean_hist += hist.flatten().astype(np.int64)

        # Finalize noisy metrics
        if noisy_pixels > 0:
            mean = noisy_sum / noisy_pixels
            var = (noisy_sq_sum / noisy_pixels) - (mean ** 2)
            std = np.sqrt(max(0.0, var))
            stats["noisy_stats"]["mean"] = float(mean)
            stats["noisy_stats"]["std"] = float(std)
        stats["noisy_stats"]["intensity_histogram_256"] = noisy_hist.tolist()

        # Finalize clean metrics
        if clean_pixels > 0:
            mean = clean_sum / clean_pixels
            var = (clean_sq_sum / clean_pixels) - (mean ** 2)
            std = np.sqrt(max(0.0, var))
            stats["clean_stats"]["mean"] = float(mean)
            stats["clean_stats"]["std"] = float(std)
        stats["clean_stats"]["intensity_histogram_256"] = clean_hist.tolist()

        # Finalize dimension metrics
        if len(widths) > 0:
            stats["dimensions"]["avg_width"] = float(np.mean(widths))
            stats["dimensions"]["avg_height"] = float(np.mean(heights))
            stats["dimensions"]["min_width"] = int(np.min(widths))
            stats["dimensions"]["max_width"] = int(np.max(widths))
            stats["dimensions"]["min_height"] = int(np.min(heights))
            stats["dimensions"]["max_height"] = int(np.max(heights))

        stats["formats"] = formats
        stats["datasets_summary"] = {k: {"pairs_count": v} for k, v in datasets_counts.items()}

        # Build splits summary
        for split_name, s_info in splits_data.items():
            split_stats = {"pairs_count": s_info["pairs_count"]}
            
            # Noisy split metrics
            npix = s_info["noisy_pixels"]
            if npix > 0:
                s_mean = s_info["noisy_sum"] / npix
                s_var = (s_info["noisy_sq_sum"] / npix) - (s_mean ** 2)
                split_stats["noisy_mean"] = float(s_mean)
                split_stats["noisy_std"] = float(np.sqrt(max(0.0, s_var)))
                
            # Clean split metrics
            cpix = s_info["clean_pixels"]
            if cpix > 0:
                s_mean = s_info["clean_sum"] / cpix
                s_var = (s_info["clean_sq_sum"] / cpix) - (s_mean ** 2)
                split_stats["clean_mean"] = float(s_mean)
                split_stats["clean_std"] = float(np.sqrt(max(0.0, s_var)))
                
            stats["splits_summary"][split_name] = split_stats

        logger.info("Dataset statistics successfully compiled.")
        return stats

    def save_statistics(self, output_path: Path) -> None:
        """Saves stats payload as dataset_statistics.json.

        Args:
            output_path (Path): Destination JSON file.
        """
        stats = self.generate_statistics()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=4)
        logger.info(f"Saved dataset statistics parameters to: {output_path.absolute()}")
