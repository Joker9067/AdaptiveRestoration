"""
Synthetic Dataset Generator orchestrator module for the Semiconductor Image Restoration System (Module 5).
Manages generating/loading clean source images, applying sequential degradations,
running optional multiprocessing, generating UUID metadata sidecars, creating
difference heatmaps, and compiling degradation_report.json summaries.
"""

import os
import time
import uuid
import json
import logging
import datetime
import multiprocessing
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import cv2
import numpy as np

from dataset_manager.config import PipelineConfig
from dataset_manager.profiler import DatasetProfiler
from synthetic_generator.pipeline import DegradationPipeline

logger = logging.getLogger(__name__)


def create_mock_pattern(pattern_type: str, size: Tuple[int, int] = (512, 512)) -> np.ndarray:
    """Generates standard semiconductor calibration patterns dynamically to avoid placeholders.

    Args:
        pattern_type (str): 'grid', 'circles', or 'siemens_star'.
        size (Tuple[int, int]): Image dimensions.

    Returns:
        np.ndarray: Grayscale pattern image (uint8).
    """
    h, w = size
    img = np.full((h, w), 50, dtype=np.uint8)  # Silicon substrate dark background

    if pattern_type == "grid":
        step = 40
        for x in range(step, w, step):
            cv2.line(img, (x, 0), (x, h), 200, thickness=6)
        for y in range(step, h, step):
            cv2.line(img, (0, y), (w, y), 200, thickness=6)
        for x in range(step, w, step):
            for y in range(step, h, step):
                cv2.circle(img, (x, y), 8, 230, -1)

    elif pattern_type == "circles":
        cx, cy = w // 2, h // 2
        for r in range(40, min(w, h) // 2 - 20, 50):
            cv2.circle(img, (cx, cy), r, 170, thickness=10)
            cv2.circle(img, (cx, cy), r + 15, 210, thickness=3)

    elif pattern_type == "siemens_star":
        cx, cy = w // 2, h // 2
        num_spokes = 24
        angle_step = 360 / num_spokes
        for i in range(num_spokes):
            angle = i * angle_step
            rad = np.deg2rad(angle)
            x2 = int(cx + np.cos(rad) * max(w, h))
            y2 = int(cy + np.sin(rad) * max(w, h))
            cv2.line(img, (cx, cy), (x2, y2), 220, thickness=8)

    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


def process_single_sample(args: Tuple[int, Path, Path, Path, str, Dict[str, Any], int, Tuple[int, int]]) -> Optional[Dict[str, Any]]:
    """Standalone target task function for multiprocessing to avoid pickling issues.

    Args:
        args: Tuple containing:
            - sample_idx: Index of the sample.
            - clean_img_path: Path to clean source image.
            - output_dir: Path to save outputs.
            - reports_dir: Path to save previews.
            - preset_name: Degradation preset name.
            - custom_pipeline: Pipeline specifications override.
            - base_seed: Global seed.
            - target_size: Preprocessor sizing.

    Returns:
        Optional[Dict[str, Any]]: Summary dictionary of the generated sample parameters.
    """
    sample_idx, clean_img_path, output_dir, reports_dir, preset_name, custom_pipeline, base_seed, target_size = args
    
    try:
        # 1. Initialize deterministic random state for this sample
        sample_seed = int(base_seed + sample_idx)
        state = np.random.RandomState(sample_seed)

        # 2. Load clean image
        clean_img = cv2.imread(str(clean_img_path), cv2.IMREAD_GRAYSCALE)
        if clean_img is None:
            return None

        # 3. Apply degradation pipeline
        pipeline = DegradationPipeline(preset_name=preset_name, custom_pipeline=custom_pipeline)
        degraded_img, metadata = pipeline.run(clean_img, state)

        # 4. Generate unique identity UUID
        sample_uuid = str(uuid.uuid4())
        timestamp = datetime.datetime.now().isoformat()

        # Filename outputs matching "Synthetic" registry pattern
        noisy_filename = f"synthetic_{sample_idx:05d}_noisy.png"
        gt_filename = f"synthetic_{sample_idx:05d}_gt.png"

        noisy_path = output_dir / noisy_filename
        gt_path = output_dir / gt_filename

        # Write image files
        cv2.imwrite(str(noisy_path), degraded_img)
        cv2.imwrite(str(gt_path), clean_img)

        # 5. Generate side-by-side difference preview heatmap panel
        # Diff is calculated on [0, 255] float arrays
        diff = np.abs(clean_img.astype(np.float32) - degraded_img.astype(np.float32))
        # Scale to emphasize differences visually
        diff_norm = np.clip(diff * 2.0, 0, 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)

        # Convert clean and degraded to 3-channel for concat
        clean_color = cv2.cvtColor(clean_img, cv2.COLOR_GRAY2BGR)
        degraded_color = cv2.cvtColor(degraded_img, cv2.COLOR_GRAY2BGR)

        # Concat horizontally
        preview_panel = np.hstack([clean_color, degraded_color, heatmap])
        
        # Add labels to preview
        cv2.putText(preview_panel, "Clean Target", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(preview_panel, "Degraded Input", (clean_img.shape[1] + 15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(preview_panel, "Diff Heatmap (x2)", (2 * clean_img.shape[1] + 15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        preview_dir = reports_dir / "synthetic_preview"
        preview_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(preview_dir / f"preview_{sample_uuid}.png"), preview_panel)

        # 6. Extract metrology parameters using profiler static checks
        noisy_entropy = DatasetProfiler._compute_entropy(degraded_img)
        # Fetch edge features: Canny density, orient hist, grad
        edge_density, _, grad_noisy = DatasetProfiler._compute_edge_features(degraded_img)
        # Texture complexity: Laplacian var, average local std
        texture_score, _ = DatasetProfiler._compute_texture_complexity(degraded_img)

        # Retrieve ground truth edge features to estimate blur reduction
        _, _, grad_gt = DatasetProfiler._compute_edge_features(clean_img)
        blur_level = float(max(0.0, (grad_gt - grad_noisy) / max(1e-6, grad_gt)))

        # Find noise level (std of diff)
        noise_level = float(np.std(clean_img.astype(np.float64) - degraded_img.astype(np.float64)) / 255.0)

        # Build payload record
        sample_meta = {
            "uuid": sample_uuid,
            "sample_index": sample_idx,
            "timestamp": timestamp,
            "seed": sample_seed,
            "source_image": clean_img_path.name,
            "noisy_image": noisy_filename,
            "gt_image": gt_filename,
            "applied_sequence": metadata["applied_sequence"],
            "parameters": metadata["parameters"],
            "severity_score": metadata["severity_score"],
            "severity_level": metadata["severity_level"],
            "metrology": {
                "noise_level": noise_level,
                "blur_level": blur_level,
                "entropy": noisy_entropy,
                "edge_density": edge_density,
                "texture_score": texture_score
            }
        }

        # Write individual UUID sidecar file for reproducibility
        sidecar_path = output_dir / f"synthetic_{sample_idx:05d}_meta.json"
        with open(sidecar_path, "w", encoding="utf-8") as f:
            json.dump(sample_meta, f, indent=4)

        return sample_meta

    except Exception as e:
        logger.error(f"Failed processing synthetic sample {sample_idx}: {e}")
        return None


class SyntheticDatasetGenerator:
    """Manages the generation of degraded/clean dataset pairs using the presets pipeline."""

    def __init__(self, config: PipelineConfig):
        """Initializes with global settings.

        Args:
            config (PipelineConfig): Global configuration instance.
        """
        self.config = config
        self.sg_cfg = config.synthetic_generator
        self.clean_dir = Path(self.sg_cfg.clean_source_dir)
        self.output_dir = Path(self.sg_cfg.output_dir)
        self.reports_dir = Path(config.paths.reports_dir)

    def _prepare_clean_sources(self) -> List[Path]:
        """Scans clean directory for images.

        Generates mock patterns if the directory is missing or empty.

        Returns:
            List[Path]: List of absolute paths to clean source images.
        """
        self.clean_dir.mkdir(parents=True, exist_ok=True)
        
        valid_extensions = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}
        files = [
            f for f in self.clean_dir.glob("*")
            if f.is_file() and f.suffix.lower() in valid_extensions
        ]

        if not files:
            logger.info(f"Clean source directory '{self.clean_dir}' holds no files. Creating mock calibration patterns...")
            patterns = ["grid", "circles", "siemens_star"]
            for pat in patterns:
                img = create_mock_pattern(pat, (512, 512))
                out_path = self.clean_dir / f"pattern_{pat}.png"
                cv2.imwrite(str(out_path), img)
                files.append(out_path)
            logger.info("Mock calibration patterns generated successfully.")

        return sorted(files)

    def generate(self) -> List[Dict[str, Any]]:
        """Orchestrates degradation processing using optional multiprocessing.

        Returns:
            List[Dict[str, Any]]: List of metadata entries for generated samples.
        """
        if not self.sg_cfg.enabled:
            logger.info("Synthetic Dataset Generator is disabled in config.")
            return []

        logger.info("Initializing Synthetic Dataset Generation process...")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # 1. Fetch clean sources
        clean_sources = self._prepare_clean_sources()
        if not clean_sources:
            raise RuntimeError("No clean source images resolved. Aborting generation.")

        # 2. Build task list args
        tasks = []
        for i in range(self.sg_cfg.num_samples):
            # Cycle through source images
            source_img = clean_sources[i % len(clean_sources)]
            tasks.append((
                i,
                source_img,
                self.output_dir,
                self.reports_dir,
                self.sg_cfg.preset,
                self.sg_cfg.pipeline,
                self.sg_cfg.seed,
                (self.config.preprocessing.resize.height, self.config.preprocessing.resize.width)
            ))

        generated_metadata: List[Dict[str, Any]] = []

        # 3. Optional Multiprocessing
        use_mp = self.sg_cfg.multiprocessing and self.sg_cfg.num_workers > 1
        
        if use_mp:
            logger.info(f"Spawning task multiprocessing pool with {self.sg_cfg.num_workers} workers...")
            try:
                # Using multiprocessing pool with map
                with multiprocessing.Pool(processes=self.sg_cfg.num_workers) as pool:
                    results = pool.map(process_single_sample, tasks)
                
                # Filter out failures
                generated_metadata = [r for r in results if r is not None]
                logger.info(f"Multiprocessing complete. Generated {len(generated_metadata)} samples.")
            except Exception as e:
                logger.warning(f"Multiprocessing failed: {e}. Falling back to single-threaded mode.")
                use_mp = False

        if not use_mp:
            logger.info("Executing generation in single-threaded mode...")
            for task_args in tasks:
                res = process_single_sample(task_args)
                if res:
                    generated_metadata.append(res)
            logger.info(f"Single-threaded execution complete. Generated {len(generated_metadata)} samples.")

        # 4. Generate summary report & save degradation_report.json
        if generated_metadata:
            self._compile_summary_report(generated_metadata)

        return generated_metadata

    def _compile_summary_report(self, metadata: List[Dict[str, Any]]) -> None:
        """Calculates degradation distributions, average parameters, and compiles degradation_report.json."""
        logger.info("Compiling overall degradation statistics report...")
        
        degradation_counts = {}
        severity_counts = {"Easy": 0, "Medium": 0, "Hard": 0, "Extreme": 0}
        source_counts = {}
        
        total_samples = len(metadata)
        noise_level_sum = 0.0
        blur_level_sum = 0.0
        entropy_sum = 0.0
        edge_sum = 0.0
        texture_sum = 0.0

        for m in metadata:
            # 1. Count active degradation sequences
            for op in m["applied_sequence"]:
                degradation_counts[op] = degradation_counts.get(op, 0) + 1
            
            # 2. Count severity
            level = m["severity_level"]
            severity_counts[level] = severity_counts.get(level, 0) + 1

            # 3. Source counts
            src = m["source_image"]
            source_counts[src] = source_counts.get(src, 0) + 1

            # 4. Metrology accumulators
            met = m["metrology"]
            noise_level_sum += met["noise_level"]
            blur_level_sum += met["blur_level"]
            entropy_sum += met["entropy"]
            edge_sum += met["edge_density"]
            texture_sum += met["texture_score"]

        # Formulate average distributions
        averages = {
            "avg_noise_level": noise_level_sum / total_samples,
            "avg_blur_level": blur_level_sum / total_samples,
            "avg_entropy": entropy_sum / total_samples,
            "avg_edge_density": edge_sum / total_samples,
            "avg_texture_complexity": texture_sum / total_samples
        }

        # Percentage frequency values
        degradation_frequencies = {k: float(v) / total_samples for k, v in degradation_counts.items()}
        severity_frequencies = {k: float(v) / total_samples for k, v in severity_counts.items()}

        report = {
            "total_samples_generated": total_samples,
            "generation_preset": self.sg_cfg.preset,
            "seed": self.sg_cfg.seed,
            "timestamp": datetime.datetime.now().isoformat(),
            "degradation_frequency": degradation_frequencies,
            "severity_distribution": severity_frequencies,
            "average_metrology_scores": averages,
            "samples_per_source": source_counts,
            "samples_metadata": metadata
        }

        report_path = self.output_dir / "degradation_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)
        logger.info(f"Saved synthetic degradation report to {report_path.absolute()}")
