"""
Demonstration and execution pipeline for the Semiconductor Image Dataset System.
1. Generates highly realistic mock semiconductor calibration targets to avoid placeholder code.
   - Simulates Poisson shot noise, Gaussian electronics noise, e-beam blur, and charge artifacts.
   - Packages a 3D SBF-SEM volume stack.
2. Automates raw directory setups (SEM, NIST, SIDD, SBF_SEM, Synthetic).
3. Executes the format conversion, stack slicing, pairing, and division splits.
4. Generates the metadata spreadsheet.
5. Examines data loader iterations on PyTorch tensors.
6. Saves diagnostic side-by-side plots, Gray intensity histograms, and global metrics profiles.
"""

import logging
import os
import shutil
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

# Import dataset manager
from dataset_manager import (
    PipelineConfig,
    DatasetManager,
    create_dataloaders,
    DatasetVisualizer,
    setup_pipeline_logging,
    calculate_psnr,
    calculate_ssim,
    calculate_relative_snr,
)


logger = logging.getLogger("run_pipeline")


def create_calibration_pattern(pattern_type: str = "grid", size: Tuple[int, int] = (512, 512)) -> np.ndarray:
    """Generates deterministic semiconductor-like calibration targets.

    Args:
        pattern_type (str): 'grid', 'circles', or 'siemens_star'.
        size (Tuple[int, int]): Image dimensions.

    Returns:
        np.ndarray: Grayscale pattern image (uint8).
    """
    h, w = size
    img = np.full((h, w), 50, dtype=np.uint8) # Dark background (silicon substrate)

    if pattern_type == "grid":
        # Simulate circuit cross-section grid traces (lines representing SiO2 or Copper lines)
        step = 40
        for x in range(step, w, step):
            cv2.line(img, (x, 0), (x, h), 200, thickness=8)
        for y in range(step, h, step):
            cv2.line(img, (0, y), (w, y), 200, thickness=8)
        # Add tiny contact vias
        for x in range(step, w, step):
            for y in range(step, h, step):
                cv2.circle(img, (x, y), 10, 240, -1)

    elif pattern_type == "circles":
        # Concentric circular transistor features or contact pads
        cx, cy = w // 2, h // 2
        for r in range(40, min(w, h) // 2 - 20, 60):
            cv2.circle(img, (cx, cy), r, 180, thickness=12)
            cv2.circle(img, (cx, cy), r + 20, 220, thickness=4)

    elif pattern_type == "siemens_star":
        # Standard metrology resolution evaluation star
        cx, cy = w // 2, h // 2
        num_spokes = 24
        angle_step = 360 / num_spokes
        for i in range(num_spokes):
            angle = i * angle_step
            rad = np.deg2rad(angle)
            x2 = int(cx + np.cos(rad) * max(w, h))
            y2 = int(cy + np.sin(rad) * max(w, h))
            cv2.line(img, (cx, cy), (x2, y2), 220, thickness=10)

    # Apply moderate post-blur to simulate optical/beam transfer functions
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return img


def simulate_sem_noise(
    clean_image: np.ndarray,
    poisson_scale: float = 15.0,
    gaussian_sigma: float = 0.04,
    charging_lines: bool = True,
    beam_blur: bool = True,
    blur_sigma: float = 1.2,
) -> np.ndarray:
    """Applies realistic Scanning Electron Microscope degradations:

    1. Electron beam focus blur.
    2. Electron shot noise (Poisson distribution).
    3. Backscatter detector electronic read noise (Gaussian distribution).
    4. Substrate charging jitter (horizontal bright streak scanlines).

    Args:
        clean_image (np.ndarray): Original image array (uint8).
        poisson_scale (float): Scale for Poisson noise.
        gaussian_sigma (float): Sigma for Gaussian noise relative to [0-1] range.
        charging_lines (bool): Enforce scanline streak artifacts.
        beam_blur (bool): Apply defocus blur.
        blur_sigma (float): Lens blur sigma.

    Returns:
        np.ndarray: Blurrred, noisy SEM simulator output.
    """
    img_float = clean_image.astype(np.float64) / 255.0

    # 1. Defocus Blur
    if beam_blur and blur_sigma > 0:
        k_size = int(2 * round(3 * blur_sigma) + 1)
        img_float = cv2.GaussianBlur(img_float, (k_size, k_size), blur_sigma)

    # 2. Electron Shot Noise (Poisson)
    # Electrons are random events, variance depends on count
    # Poisson(Intensity * Scale) / Scale
    if poisson_scale > 0:
        # Scale determines average electron count per pixel
        scaled = img_float * poisson_scale
        # Clip to prevent nan issues
        scaled = np.clip(scaled, 0.001, None)
        noisy_poisson = np.random.poisson(scaled) / poisson_scale
        img_float = noisy_poisson

    # 3. Readout electronic noise (Gaussian)
    if gaussian_sigma > 0:
        noise_g = np.random.normal(0, gaussian_sigma, img_float.shape)
        img_float = img_float + noise_g

    # 4. Scanning Charging streaks
    # Electron building-up on insulative substrates creates bright horizontal raster streaks
    if charging_lines:
        h, w = img_float.shape[:2]
        num_streaks = np.random.randint(1, 4)
        for _ in range(num_streaks):
            y = np.random.randint(0, h)
            strength = np.random.uniform(0.1, 0.3)
            # Create a localized vertical row gradient represent raster charging glow
            glow_height = np.random.randint(2, 6)
            for offset in range(-glow_height, glow_height):
                if 0 <= y + offset < h:
                    falloff = 1.0 - (abs(offset) / glow_height)
                    img_float[y + offset, :] += strength * falloff

    # Post processing: normalizations and cast back to uint8
    img_float = np.clip(img_float, 0.0, 1.0)
    return (img_float * 255.0).astype(np.uint8)


def generate_mock_datasets(raw_dir: Path) -> None:
    """Creates the folders and writes beautiful mock dataset patterns with SEM artifacts.

    Matches: SEM, NIST, SIDD, SBF_SEM, Synthetic.

    Args:
        raw_dir (Path): Destination location where directories are built.
    """
    logger.info(f"Generating synthetic mock datasets under {raw_dir} for pipeline integration test...")
    
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------
    # 1. Dataset type: SEM
    # Subfolder scheme: parallel 'clean' and 'noisy' folders
    # ----------------------------------------------------
    sem_dir = raw_dir / "SEM"
    for split in ["clean", "noisy"]:
        (sem_dir / split).mkdir(parents=True, exist_ok=True)
    
    patterns = ["grid", "circles", "siemens_star"]
    for i, pat in enumerate(patterns):
        clean = create_calibration_pattern(pat, (512, 512))
        # Noisy has high Poisson noise (scalep=8.0) and severe charging streaks
        noisy = simulate_sem_noise(clean, poisson_scale=8.0, gaussian_sigma=0.06, charging_lines=True)
        
        cv2.imwrite(str(sem_dir / "clean" / f"track_{i:02d}.jpg"), clean)
        cv2.imwrite(str(sem_dir / "noisy" / f"track_{i:02d}.jpg"), noisy)

    # ----------------------------------------------------
    # 2. Dataset type: NIST
    # Flat namespace format: matching suffixes
    # ----------------------------------------------------
    nist_dir = raw_dir / "NIST_SEM"
    nist_dir.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        pat = patterns[i % len(patterns)]
        clean = create_calibration_pattern(pat, (640, 480)) # Diff scale aspect ratio
        noisy = simulate_sem_noise(clean, poisson_scale=12.0, gaussian_sigma=0.03, charging_lines=False)
        
        cv2.imwrite(str(nist_dir / f"nist_sample_{i:03d}_clean.png"), clean)
        cv2.imwrite(str(nist_dir / f"nist_sample_{i:03d}_noisy.png"), noisy)

    # ----------------------------------------------------
    # 3. Dataset type: SIDD
    # Subfolder format: parallel 'gt' vs 'noisy'
    # ----------------------------------------------------
    sidd_dir = raw_dir / "SIDD_Pairs"
    (sidd_dir / "gt").mkdir(parents=True, exist_ok=True)
    (sidd_dir / "noisy").mkdir(parents=True, exist_ok=True)
    for i in range(2):
        pat = patterns[(i + 1) % len(patterns)]
        clean = create_calibration_pattern(pat, (512, 512))
        noisy = simulate_sem_noise(clean, poisson_scale=10.0, gaussian_sigma=0.05, charging_lines=True)
        
        cv2.imwrite(str(sidd_dir / "gt" / f"sidd_ref_{i}.bmp"), clean) # Test BMP reads
        cv2.imwrite(str(sidd_dir / "noisy" / f"sidd_ref_{i}.bmp"), noisy)

    # ----------------------------------------------------
    # 4. Dataset type: SBF_SEM
    # Uses 3D Multi-page TIFF stacks!
    # ----------------------------------------------------
    sbf_dir = raw_dir / "SBF_SEM_Volumes"
    sbf_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate 5 slices representing a moving 3D structures (shifting circle coords)
    clean_slices: List[Image.Image] = []
    noisy_slices: List[Image.Image] = []
    
    for slice_idx in range(5):
        # Background
        c_layer = np.full((512, 512), 40, dtype=np.uint8)
        # Simulating cross section trace moving in 3D: circles offset dynamically
        cx = 256 + int(50 * np.sin(slice_idx * np.pi / 4))
        cy = 256 + int(50 * np.cos(slice_idx * np.pi / 4))
        cv2.circle(c_layer, (cx, cy), 80, 200, thickness=15)
        c_layer = cv2.GaussianBlur(c_layer, (5,5), 0)
        
        n_layer = simulate_sem_noise(c_layer, poisson_scale=15.0, gaussian_sigma=0.04, charging_lines=True)
        
        clean_slices.append(Image.fromarray(c_layer))
        noisy_slices.append(Image.fromarray(n_layer))

    # Save as 3D TIFF stack volumes
    clean_slices[0].save(
        sbf_dir / "volume_clean_ref.tiff",
        save_all=True,
        append_images=clean_slices[1:],
        format="TIFF",
    )
    noisy_slices[0].save(
        sbf_dir / "volume_noisy_input.tiff",
        save_all=True,
        append_images=noisy_slices[1:],
        format="TIFF",
    )

    # ----------------------------------------------------
    # 5. Dataset type: Synthetic
    # Normal naming style, flat directory
    # ----------------------------------------------------
    syn_dir = raw_dir / "Synthetic_Tuning"
    syn_dir.mkdir(parents=True, exist_ok=True)
    for i in range(2):
        clean = create_calibration_pattern("siemens_star", (512, 512))
        noisy = simulate_sem_noise(clean, poisson_scale=20.0, gaussian_sigma=0.02, charging_lines=False)
        
        cv2.imwrite(str(syn_dir / f"syn_calib_{i}_gt.png"), clean)
        cv2.imwrite(str(syn_dir / f"syn_calib_{i}_noisy.png"), noisy)

    logger.info("Mock raw datasets successfully created!")


def main() -> None:
    """Executes the validation and preprocessing pipeline end to end."""
    # 1. Load config
    config_path = Path("./config.yaml")
    config = PipelineConfig.load_from_yaml(config_path)

    # 2. Setup Logging
    # Redirects logging to file and screen stdout
    setup_pipeline_logging(config.paths.log_file, level=logging.INFO)
    logger.info("Starting Dataset Management System testing framework...")

    workspace_dir = Path(".").resolve()
    logger.info(f"Workspace directory resolved: {workspace_dir}")

    # Make absolute paths for testing robustness
    raw_path = Path(config.paths.raw_dir).resolve()
    processed_path = Path(config.paths.processed_dir).resolve()

    # 3. Generate mock inputs
    generate_mock_datasets(raw_path)

    # Trigger Synthetic Dataset Generation (Module 5) if enabled
    if config.synthetic_generator.enabled:
        from synthetic_generator.generator import SyntheticDatasetGenerator
        logger.info("Synthetic Dataset Generator is enabled. Generating synthetic image pairs...")
        generator = SyntheticDatasetGenerator(config)
        generator.generate()

    # 4. Initialize DatasetManager and execute
    # Automatically detects availability, downloads remote files if configured/missing,
    # and processes, slices, standardizes, pairs, splits, and writes metadata.
    manager = DatasetManager(config)
    records = manager.prepare_and_process_all()

    if not records:
        logger.error("Preprocessing pipeline returned 0 records. Pipeline failed.")
        return

    # Verify loading back metadata
    metadata_csv_path = manager.metadata_csv_path
    loaded_df = manager.metadata_mgr.load_metadata()
    logger.info(f"Successfully validated metadata.csv! Loaded shape: {loaded_df.shape}")


    # 6. Verify PyTorch custom dataset and loaders
    train_loader, val_loader, test_loader = create_dataloaders(
        metadata_csv_path=metadata_csv_path,
        config=config,
        base_dir=workspace_dir,
    )

    # Iterate a single batch to verify shapes and tensor conversions
    if train_loader:
        logger.info("Testing Train DataLoader iterations...")
        for batch_idx, (inputs_t, targets_t) in enumerate(train_loader):
            logger.info(
                f"Batch {batch_idx + 1} loaded. "
                f"Input tensor: shape={list(inputs_t.shape)}, dtype={inputs_t.dtype}, device={inputs_t.device} "
                f"Target tensor: shape={list(targets_t.shape)}, dtype={targets_t.dtype}, device={targets_t.device}"
            )
            # Assert normalizations are respected
            assert inputs_t.min() >= 0.0 and inputs_t.max() <= 1.0, "Input normalization error"
            assert targets_t.min() >= 0.0 and targets_t.max() <= 1.0, "GT normalization error"
            break # Just testing the loading trigger

    # 7. Quality analysis visualizations and reports
    logger.info("Initializing visualizers and plotting profiles...")
    visualizer = DatasetVisualizer(workspace_root=workspace_dir)

    # Output random samples side-by-side matches and distributions
    vis_out_dir = Path(config.paths.reports_dir) / "visualizations"
    visualizer.visualize_random_samples(metadata_csv_path, vis_out_dir, num_samples=3)

    # Compile global distributions (violin/histogram graphs of MSE/PSNR/SSIM/SNR across splits)
    dist_plot_path = vis_out_dir / "dataset_metrics_distributions.png"
    visualizer.compile_global_metric_distributions(metadata_csv_path, dist_plot_path)

    # 8. Report final metrics summarizing database quality logs
    logger.info("Computing metrics summaries across preprocessed splits...")
    metrics_csv = vis_out_dir / "dataset_metrics_distributions.csv"
    if metrics_csv.exists():
        rdf = pd.read_csv(metrics_csv)
        summary = rdf.groupby(["Split", "Dataset"])[["MSE", "PSNR (dB)", "SSIM", "SNR (dB)"]].mean()
        print("\n" + "="*80)
        print("                PREPROCESSED DATASET METRIC LOG SUMMARIES")
        print("="*80)
        print(summary.to_string())
        print("="*80 + "\n")

    logger.info("Semiconductor Dataset Management System execution completed successfully!")


if __name__ == "__main__":
    # Workaround for windows multiprocessing freeze on double invoke
    main()
