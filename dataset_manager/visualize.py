"""
Visualization module for the Semiconductor Image Dataset System.
Provides formatting plotting utilities:
1. Pair visualization: side-by-side input (noisy) and ground truth (clean).
2. Intensity Histogram: overlaps pixel gray distributions to analyze noise characteristics.
3. Metric distribution plots: visualizes overall PSNR/SSIM/SNR across splits.
"""

import logging
import random
from pathlib import Path
from typing import Optional, Union, Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataset_manager.metadata import MetadataManager
from dataset_manager.utils import calculate_mse, calculate_psnr, calculate_ssim, calculate_relative_snr

logger = logging.getLogger(__name__)

# Standardize plot theme for a premium scientific look
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 14,
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
})

class DatasetVisualizer:
    """Provides high-quality visualization assets for semiconductor image restoration quality control."""

    def __init__(self, workspace_root: Path):
        """Initializes with the path to the workspace root directory.

        Args:
            workspace_root (Path): Path to locate relative file positions.
        """
        self.workspace_root = Path(workspace_root)

    def load_pair(self, row: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Loads clean and noisy image arrays matching a metadata row.

        Args:
            row (pd.Series): DataFrame row.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (noisy_input, ground_truth) arrays.
        """
        inp_path = self.workspace_root / row["input_path"]
        gt_path = self.workspace_root / row["ground_truth_path"]

        inp_img = cv2.imread(str(inp_path), cv2.IMREAD_GRAYSCALE)
        gt_img = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)

        if inp_img is None or gt_img is None:
            raise FileNotFoundError(f"Failed to load image pair: {inp_path} / {gt_path}")

        return inp_img, gt_img

    def plot_pair_comparison(
        self,
        row: pd.Series,
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """Generates side-by-side plot comparing noisy input and clean ground truth.

        Displays dimensions, MSE, PSNR, SSIM, and local SNR metrics.

        Args:
            row (pd.Series): Metadata record row.
            save_path (Path, optional): Destination image path.

        Returns:
            plt.Figure: Generated figure handle.
        """
        inp_img, gt_img = self.load_pair(row)

        # Compute metrics
        mse = calculate_mse(gt_img, inp_img)
        psnr = calculate_psnr(gt_img, inp_img)
        ssim = calculate_ssim(gt_img, inp_img)
        snr = calculate_relative_snr(gt_img, inp_img)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Plot Noisy Input
        axes[0].imshow(inp_img, cmap="gray", vmin=0, vmax=255)
        axes[0].set_title(f"Noisy Input ({row['dataset_name']})\nType: {row['image_type']}")
        axes[0].set_xlabel(f"Width: {row['width']}px | Height: {row['height']}px")
        axes[0].axis("on")
        
        # Plot Ground Truth
        axes[1].imshow(gt_img, cmap="gray", vmin=0, vmax=255)
        axes[1].set_title("Ground Truth (Clean Reference)")
        axes[1].set_xlabel("Target Dimensions")
        axes[1].axis("on")

        # Overlay quantitative metrics
        fig.suptitle(
            f"Image restoration Pair Comparison: {row['image_name']}\n"
            f"PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f} | MSE: {mse:.2f} | SNR: {snr:.2f} dB",
            y=0.98,
        )
        
        plt.tight_layout()
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.debug(f"Saved side-by-side visual comparison to {save_path}")
            
        return fig

    def plot_gray_value_histogram(
        self,
        row: pd.Series,
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """Plots overlapping pixel intensity distributions for the image pair.

        Analyzes noise skewness, variance, and mean shifts.

        Args:
            row (pd.Series): Metadata record row.
            save_path (Path, optional): Destination image path.

        Returns:
            plt.Figure: Generated figure handle.
        """
        inp_img, gt_img = self.load_pair(row)

        fig, (ax_img, ax_hist) = plt.subplots(1, 2, figsize=(11, 4.5))

        # Show side by side difference image as pixel visualizer
        diff = cv2.absdiff(gt_img, inp_img)
        ax_img.imshow(diff, cmap="hot")
        ax_img.set_title("Noise Pattern (Residual Difference)")
        ax_img.axis("off")
        
        # Plot Histograms
        ax_hist.hist(
            gt_img.ravel(),
            bins=256,
            range=(0, 256),
            color="#22c55e",
            alpha=0.6,
            label="Ground Truth (Clean)",
            density=True,
        )
        ax_hist.hist(
            inp_img.ravel(),
            bins=256,
            range=(0, 256),
            color="#ef4444",
            alpha=0.45,
            label="Input (Noisy)",
            density=True,
        )
        
        ax_hist.set_title("Pixel Intensity Frequency Histograms")
        ax_hist.set_xlabel("Gray Level (0 - 255)")
        ax_hist.set_ylabel("Probability Density")
        ax_hist.legend(loc="upper right")
        
        # Add basic stats Box
        gt_mean, gt_std = np.mean(gt_img), np.std(gt_img)
        inp_mean, inp_std = np.mean(inp_img), np.std(inp_img)
        
        stats_text = (
            f"Ground Truth:\n"
            f"  Mean: {gt_mean:.1f}\n"
            f"  Std:  {gt_std:.1f}\n"
            f"Noisy Input:\n"
            f"  Mean: {inp_mean:.1f}\n"
            f"  Std:  {inp_std:.1f}"
        )
        ax_hist.text(
            0.05,
            0.95,
            stats_text,
            transform=ax_hist.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8, edgecolor="#d1d5db"),
            fontfamily="monospace",
        )

        fig.suptitle(f"Noise Profile Histogram: {row['image_name']}", y=0.98)
        plt.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.debug(f"Saved intensity distribution histogram to {save_path}")

        return fig

    def compile_global_metric_distributions(
        self,
        metadata_csv_path: Path,
        save_path: Path,
    ) -> None:
        """Calculates PSNR, SSIM, MSE, and SNR for all paired images and plots distributions.

        Helps profile the whole dataset quality in train/val/test splits.

        Args:
            metadata_csv_path (Path): Path to metadata.csv.
            save_path (Path): Image output path.
        """
        meta_mgr = MetadataManager(metadata_csv_path)
        try:
            df = meta_mgr.load_metadata()
        except Exception as e:
            logger.error(f"Cannot compile global metrics distributions: {e}")
            return

        if len(df) == 0:
            logger.warning("Empty metadata CSV file. Skipping global metrics compilation.")
            return

        logger.info("Computing quality metrics for all dataset images (profile distribution)...")
        
        psnrs, ssims, mses, snrs = [], [], [], []
        splits = []
        datasets = []

        for idx, row in df.iterrows():
            try:
                inp_img, gt_img = self.load_pair(row)
                mses.append(calculate_mse(gt_img, inp_img))
                psnrs.append(calculate_psnr(gt_img, inp_img))
                ssims.append(calculate_ssim(gt_img, inp_img))
                snrs.append(calculate_relative_snr(gt_img, inp_img))
                splits.append(row["split"])
                datasets.append(row["dataset_name"])
            except Exception as e:
                logger.error(f"Failed to process row {idx} for metrics plot: {e}")

        # Build metric dataframe
        metrics_df = pd.DataFrame({
            "Split": splits,
            "Dataset": datasets,
            "MSE": mses,
            "PSNR (dB)": psnrs,
            "SSIM": ssims,
            "SNR (dB)": snrs,
        })

        # Save metrics analysis locally as CSV
        stats_csv = save_path.with_suffix(".csv")
        metrics_df.to_csv(stats_csv, index=False)
        logger.info(f"Saved detailed numerical metrics logs to: {stats_csv}")

        # Setup 2x2 subplot distribution
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle("Global Semiconductor Image Dataset Quality & Noise Distribution Profiles", y=0.98)

        cols = ["PSNR (dB)", "SSIM", "MSE", "SNR (dB)"]
        colors = ["#3b82f6", "#10b981", "#ef4444", "#f59e0b"]

        for idx, col_name in enumerate(cols):
            ax = axes[idx // 2, idx % 2]
            
            # Simple list of split categories
            unique_splits = list(metrics_df["Split"].unique())
            
            # We plot histograms separated by split category
            for sp_idx, sp in enumerate(unique_splits):
                sp_data = metrics_df[metrics_df["Split"] == sp][col_name]
                if len(sp_data) > 0:
                    ax.hist(
                        sp_data,
                        bins=min(15, len(sp_data)),
                        alpha=0.6,
                        label=f"{sp} (mean: {np.mean(sp_data):.2f})",
                        density=False,
                    )
            
            ax.set_title(f"{col_name} Distribution")
            ax.set_xlabel(col_name)
            ax.set_ylabel("Frequency Count")
            ax.legend(loc="upper right", frameon=True)

        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved metric distribution charts to {save_path.absolute()}")

    def visualize_random_samples(
        self,
        metadata_csv_path: Path,
        output_dir: Path,
        num_samples: int = 3,
    ) -> None:
        """Selects random pairs from metadata and outputs comparative diagrams to disk.

        Args:
            metadata_csv_path (Path): Path to metadata file.
            output_dir (Path): Output directory.
            num_samples (int): Max number of samples to process.
        """
        meta_mgr = MetadataManager(metadata_csv_path)
        try:
            df = meta_mgr.load_metadata()
        except Exception:
            return

        if len(df) == 0:
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        num_samples = min(num_samples, len(df))
        
        # Pick random row indexes
        indices = random.sample(range(len(df)), num_samples)
        
        for idx in indices:
            row = df.iloc[idx]
            clean_name = Path(row["image_name"]).stem
            
            # 1. Output comparison
            comp_path = output_dir / f"{clean_name}_comparison.png"
            self.plot_pair_comparison(row, comp_path)
            
            # 2. Output histogram profile
            hist_path = output_dir / f"{clean_name}_histogram.png"
            self.plot_gray_value_histogram(row, hist_path)
            
        logger.info(f"Generated side-by-side comparison images and histograms in: {output_dir.absolute()}")
