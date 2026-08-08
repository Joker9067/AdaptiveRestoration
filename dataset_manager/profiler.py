"""
Dataset Profiler module for the Semiconductor Image Restoration System.
Analyzes restoration pair characteristics:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- Entropy (information capacity)
- Edge Density (using Canny edges) & Direction distributions
- Texture Complexity (Laplacian variance & local variance)
- Brightness Distributions (mean, std, skewness, kurtosis)
- Estimated Noise Standard Deviation
- Resolution reduction ratio
Saves profile data as dataset_profile.json, and writes metrics back to metadata.csv.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class DatasetProfiler:
    """Calculates image-level and dataset-level degradation profiles and structural signatures."""

    def __init__(self, metadata_path: Path):
        """Initializes class with processed metadata.csv path.

        Args:
            metadata_path (Path): Path to metadata.csv.
        """
        self.metadata_path = Path(metadata_path)

    @staticmethod
    def _compute_entropy(img: np.ndarray) -> float:
        """Computes the Shannon Entropy of a grayscale image."""
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist_norm = hist.flatten() / hist.sum()
        non_zero = hist_norm[hist_norm > 0]
        return float(-np.sum(non_zero * np.log2(non_zero)))

    @staticmethod
    def _compute_edge_features(img: np.ndarray) -> Tuple[float, List[float], float]:
        """Computes proportion of edge pixels and edge orientation histograms using Canny & Sobel.

        Returns:
            Tuple[float, List[float], float]:
                - Edge density (Canny pixel ratio)
                - Orientation histogram (8 bins)
                - Mean gradient strength
        """
        # 1. Edge Density
        high_thresh, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        low_thresh = 0.5 * high_thresh
        edges = cv2.Canny(img, int(low_thresh), int(high_thresh))
        edge_density = float(np.sum(edges > 0) / edges.size)

        # 2. Gradient strength & orientation histogram
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        orientation = np.arctan2(sobely, sobelx) * (180.0 / np.pi) % 360.0

        mean_grad = float(np.mean(magnitude))

        # Build 8-bin histogram of orientations weighted by gradient magnitude
        hist, _ = np.histogram(orientation, bins=8, range=(0, 360), weights=magnitude)
        total_mag = np.sum(hist)
        if total_mag > 0:
            hist_norm = (hist / total_mag).tolist()
        else:
            hist_norm = [0.0] * 8

        return edge_density, hist_norm, mean_grad

    @staticmethod
    def _compute_texture_complexity(img: np.ndarray) -> Tuple[float, float]:
        """Computes texture complexity based on the variance of Laplacian response and local variance.

        Returns:
            Tuple[float, float]: (Laplacian variance, Average Local standard deviation)
        """
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        lap_var = float(laplacian.var())

        # Average local standard deviation using a sliding window of 7x7
        img_f = img.astype(np.float64) / 255.0
        mean = cv2.blur(img_f, (7, 7))
        mean_sq = cv2.blur(img_f**2, (7, 7))
        variance = np.clip(mean_sq - mean**2, 0.0, None)
        local_std = float(np.mean(np.sqrt(variance)))

        return lap_var, local_std

    @staticmethod
    def _compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
        """Computes the Structural Similarity Index (SSIM) between two grayscale images."""
        if img1.shape != img2.shape:
            return 0.0

        x = img1.astype(np.float64) / 255.0
        y = img2.astype(np.float64) / 255.0

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = np.mean(x)
        mu_y = np.mean(y)

        sigma_x_sq = np.var(x)
        sigma_y_sq = np.var(y)
        sigma_xy = np.mean(x * y) - (mu_x * mu_y)

        num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        den = (mu_x**2 + mu_y**2 + C1) * (sigma_x_sq + sigma_y_sq + C2)

        return float(num / (den + 1e-10))

    @staticmethod
    def _compute_skew_kurtosis(img: np.ndarray) -> Tuple[float, float]:
        """Computes skewness and excess kurtosis of grayscale pixel intensities."""
        flat = img.flatten().astype(np.float32) / 255.0
        mean = np.mean(flat)
        std = np.std(flat)
        if std < 1e-6:
            return 0.0, 0.0

        diff = flat - mean
        skew = np.mean(diff ** 3) / (std ** 3)
        kurt = np.mean(diff ** 4) / (std ** 4) - 3.0  # Excess kurtosis
        return float(skew), float(kurt)

    def profile(self) -> Dict[str, Any]:
        """Calculates parameters across all images and groups metrics by dataset type.

        Also updates the metadata.csv with profile features.

        Returns:
            Dict[str, Any]: Profile payloads.
        """
        import pandas as pd
        
        profile_data = {
            "global_metrics": {
                "avg_psnr": 0.0,
                "avg_ssim": 0.0,
                "avg_entropy_noisy": 0.0,
                "avg_entropy_gt": 0.0,
                "avg_edge_density_noisy": 0.0,
                "avg_edge_density_gt": 0.0,
                "avg_texture_complexity_noisy": 0.0,
                "avg_texture_complexity_gt": 0.0,
                "avg_noise_level": 0.0,
            },
            "dataset_profiles": {},
            "image_profiles": {},
            "size_distribution": {},
        }

        if not self.metadata_path.exists():
            logger.error(f"Cannot run profiling; metadata file not found at {self.metadata_path}")
            return profile_data

        try:
            df = pd.read_csv(self.metadata_path)
        except Exception as e:
            logger.error(f"Failed to load metadata for profiling: {e}")
            return profile_data

        base_dir = self.metadata_path.parent.parent
        
        # Accumulators
        all_psnrs = []
        all_ssims = []
        all_noisy_ent = []
        all_gt_ent = []
        all_noisy_edges = []
        all_gt_edges = []
        all_noisy_tex = []
        all_gt_tex = []
        all_noise_levels = []

        ds_accumulators = {}
        sizes_seen = {}

        # We will update metadata rows with calculated profile metrics
        updated_rows = []

        for idx, row in df.iterrows():
            img_name = row.get("image_name", "UNKNOWN")
            ds_name = row.get("dataset_name", "UNKNOWN").upper()
            
            inp_rel = row.get("input_path")
            gt_rel = row.get("ground_truth_path")
            if not inp_rel or not gt_rel:
                updated_rows.append(row.to_dict())
                continue

            inp_path = base_dir.parent / Path(inp_rel)
            gt_path = base_dir.parent / Path(gt_rel)

            if not inp_path.exists() or not gt_path.exists():
                updated_rows.append(row.to_dict())
                continue

            # Load images
            inp_img = cv2.imread(str(inp_path), cv2.IMREAD_GRAYSCALE)
            gt_img = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)

            if inp_img is None or gt_img is None:
                updated_rows.append(row.to_dict())
                continue

            # 1. Sizes and Resolution Loss
            h_orig = float(row.get("original_height", inp_img.shape[0]))
            w_orig = float(row.get("original_width", inp_img.shape[1]))
            h_final, w_final = inp_img.shape[:2]
            
            res_loss = 0.0
            if h_final > 0 and w_final > 0:
                res_loss = float(1.0 - (h_final * w_final) / (h_orig * w_orig))

            size_key = f"{int(w_orig)}x{int(h_orig)}"
            sizes_seen[size_key] = sizes_seen.get(size_key, 0) + 1

            # 2. PSNR / SSIM
            mse = float(np.mean((inp_img.astype(np.float32) - gt_img.astype(np.float32))**2))
            psnr = float(20 * np.log10(255.0 / np.sqrt(mse))) if mse > 0 else 99.0
            ssim = self._compute_ssim(inp_img, gt_img)

            # 3. Noise Estimation (Standard deviation of the difference image)
            diff_img = inp_img.astype(np.float64) - gt_img.astype(np.float64)
            noise_std = float(np.std(diff_img) / 255.0)

            # 4. Entropy
            noisy_ent = self._compute_entropy(inp_img)
            gt_ent = self._compute_entropy(gt_img)

            # 5. Edge Features
            noisy_edge, noisy_edge_hist, noisy_grad = self._compute_edge_features(inp_img)
            gt_edge, gt_edge_hist, gt_grad = self._compute_edge_features(gt_img)

            # 6. Texture Complexity
            noisy_tex, noisy_loc_std = self._compute_texture_complexity(inp_img)
            gt_tex, gt_loc_std = self._compute_texture_complexity(gt_img)

            # 7. Brightness Distributions
            noisy_skew, noisy_kurt = self._compute_skew_kurtosis(inp_img)
            gt_skew, gt_kurt = self._compute_skew_kurtosis(gt_img)

            # Estimate blur level based on edge strength reduction
            blur_est = float(max(0.0, (gt_grad - noisy_grad) / max(1e-6, gt_grad)))

            # Select noise type based on dataset or properties
            noise_type = "e-beam_shot_noise"
            if "SYNTH" in ds_name:
                noise_type = "poisson_gaussian"
            elif "SIDD" in ds_name:
                noise_type = "real_sensor_noise"

            # Create row copy and update metrology attributes
            row_dict = row.to_dict()
            row_dict["noise_type"] = noise_type
            row_dict["noise_level"] = noise_std
            row_dict["blur_level"] = blur_est
            row_dict["resolution_loss"] = res_loss
            row_dict["entropy"] = noisy_ent
            row_dict["texture_score"] = noisy_tex
            row_dict["edge_density"] = noisy_edge
            updated_rows.append(row_dict)

            # Record stats
            metrics = {
                "psnr": psnr,
                "ssim": ssim,
                "noise_level": noise_std,
                "blur_level": blur_est,
                "resolution_loss": res_loss,
                "noisy_entropy": noisy_ent,
                "gt_entropy": gt_ent,
                "noisy_edge_density": noisy_edge,
                "gt_edge_density": gt_edge,
                "noisy_edge_orientation_hist_8": noisy_edge_hist,
                "gt_edge_orientation_hist_8": gt_edge_hist,
                "noisy_texture_complexity": noisy_tex,
                "gt_texture_complexity": gt_tex,
                "brightness": {
                    "noisy_mean": float(np.mean(inp_img) / 255.0),
                    "gt_mean": float(np.mean(gt_img) / 255.0),
                    "noisy_skewness": noisy_skew,
                    "noisy_kurtosis": noisy_kurt,
                    "gt_skewness": gt_skew,
                    "gt_kurtosis": gt_kurt
                }
            }

            profile_data["image_profiles"][img_name] = metrics

            # Global lists
            all_psnrs.append(psnr)
            all_ssims.append(ssim)
            all_noisy_ent.append(noisy_ent)
            all_gt_ent.append(gt_ent)
            all_noisy_edges.append(noisy_edge)
            all_gt_edges.append(gt_edge)
            all_noisy_tex.append(noisy_tex)
            all_gt_tex.append(gt_tex)
            all_noise_levels.append(noise_std)

            # Dataset-specific
            if ds_name not in ds_accumulators:
                ds_accumulators[ds_name] = {
                    "psnr": [], "ssim": [],
                    "noisy_ent": [], "gt_ent": [],
                    "noisy_edge": [], "gt_edge": [],
                    "noisy_tex": [], "gt_tex": [],
                    "noise_level": []
                }
            
            ds_accumulators[ds_name]["psnr"].append(psnr)
            ds_accumulators[ds_name]["ssim"].append(ssim)
            ds_accumulators[ds_name]["noisy_ent"].append(noisy_ent)
            ds_accumulators[ds_name]["gt_ent"].append(gt_ent)
            ds_accumulators[ds_name]["noisy_edge"].append(noisy_edge)
            ds_accumulators[ds_name]["gt_edge"].append(gt_edge)
            ds_accumulators[ds_name]["noisy_tex"].append(noisy_tex)
            ds_accumulators[ds_name]["gt_tex"].append(gt_tex)
            ds_accumulators[ds_name]["noise_level"].append(noise_std)

        # Write updated metadata back to CSV
        try:
            pd.DataFrame(updated_rows).to_csv(self.metadata_path, index=False)
            logger.info(f"Updated metadata.csv with profile attributes at {self.metadata_path}")
        except Exception as e:
            logger.error(f"Failed to update metadata.csv with profile metrics: {e}")

        # Compute averages globally
        if len(all_psnrs) > 0:
            profile_data["global_metrics"] = {
                "avg_psnr": float(np.mean(all_psnrs)),
                "avg_ssim": float(np.mean(all_ssims)),
                "avg_entropy_noisy": float(np.mean(all_noisy_ent)),
                "avg_entropy_gt": float(np.mean(all_gt_ent)),
                "avg_edge_density_noisy": float(np.mean(all_noisy_edges)),
                "avg_edge_density_gt": float(np.mean(all_gt_edges)),
                "avg_texture_complexity_noisy": float(np.mean(all_noisy_tex)),
                "avg_texture_complexity_gt": float(np.mean(all_gt_tex)),
                "avg_noise_level": float(np.mean(all_noise_levels)),
            }

        # Compute averages per dataset
        for ds_name, acc in ds_accumulators.items():
            if len(acc["psnr"]) > 0:
                profile_data["dataset_profiles"][ds_name] = {
                    "avg_psnr": float(np.mean(acc["psnr"])),
                    "avg_ssim": float(np.mean(acc["ssim"])),
                    "avg_entropy_noisy": float(np.mean(acc["noisy_ent"])),
                    "avg_entropy_gt": float(np.mean(acc["gt_ent"])),
                    "avg_edge_density_noisy": float(np.mean(acc["noisy_edge"])),
                    "avg_edge_density_gt": float(np.mean(acc["gt_edge"])),
                    "avg_texture_complexity_noisy": float(np.mean(acc["noisy_tex"])),
                    "avg_texture_complexity_gt": float(np.mean(acc["gt_tex"])),
                    "avg_noise_level": float(np.mean(acc["noise_level"])),
                    "samples_count": len(acc["psnr"]),
                }

        profile_data["size_distribution"] = sizes_seen
        logger.info(f"Profiling complete. Analyzed {len(profile_data['image_profiles'])} image pairs.")
        return profile_data

    def save_profile(self, output_path: Path) -> None:
        """Saves profiles database to output JSON.

        Args:
            output_path (Path): Path to output JSON.
        """
        profile_data = self.profile()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(profile_data, f, indent=4)
        logger.info(f"Saved dataset profile attributes database to {output_path.absolute()}")
