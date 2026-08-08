"""
Dataset Manager orchestrator module for the Semiconductor Image Restoration System.
Manages automatic location detection:
- Verifies if datasets exist locally.
- If missing, delegates downloading to registries-resolved downloaders.
- Coordinates unpacking, structuring, automatic preprocessing, splits, and metadata registration.
- Executes Validation, Statistics generation, Profiling, and Reporting in sequence.
- Stops execution if critical validation audits fail.
- Computes SHA256 dataset fingerprints and generates dataset cards and experiment templates.
"""

import logging
import hashlib
import json
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd

from dataset_manager.config import PipelineConfig, DatasetSourceConfig
from dataset_manager.preprocess import DatasetPreprocessor
from dataset_manager.metadata import MetadataManager
from dataset_manager.downloader import (
    DownloaderRegistry,
    ArchiveExtractor,
)
from dataset_manager.validator import DatasetValidator
from dataset_manager.stats import DatasetStatsGenerator
from dataset_manager.reporter import DatasetReporter
from dataset_manager.profiler import DatasetProfiler
from dataset_manager.dataset import SemiconductorDataset

logger = logging.getLogger(__name__)

class DatasetManager:
    """Core orchestrator class that manages the raw data lifecycle: download, extraction, organization, preprocessing, and registry."""

    def __init__(self, config: PipelineConfig):
        """Initializes with global settings.

        Args:
            config (PipelineConfig): Global configuration instance.
        """
        self.config = config
        self.raw_dir = Path(config.paths.raw_dir)
        self.processed_dir = Path(config.paths.processed_dir)
        self.reports_dir = Path(config.paths.reports_dir)
        self.preprocessor = DatasetPreprocessor(config)
        self.metadata_csv_path = self.processed_dir / "metadata.csv"
        self.metadata_mgr = MetadataManager(self.metadata_csv_path)
        self.cache_config_path = self.processed_dir / "cache_config.json"

    def is_dataset_locally_available(self, dataset_cfg: DatasetSourceConfig) -> bool:
        """Verifies if the dataset exists locally and contains images or stacks."""
        local_path = Path(dataset_cfg.local_path)
        if not local_path.exists():
            return False

        # Scan for images or archives
        image_extensions = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".zip", ".tar", ".gz"}
        files = [
            f for f in local_path.glob("**/*")
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        return len(files) > 0

    def download_and_extract_dataset(self, dataset_cfg: DatasetSourceConfig) -> None:
        """Chooses the correct downloader instance using DownloaderRegistry."""
        source_type = dataset_cfg.source_type.lower()
        local_path = Path(dataset_cfg.local_path)
        local_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Dataset '{dataset_cfg.name}' is missing locally. "
            f"Attempting download via registry source type '{dataset_cfg.source_type}'..."
        )

        try:
            downloader = DownloaderRegistry.get_downloader(source_type, dataset_cfg)
        except ValueError as ve:
            if source_type == "local":
                logger.info(f"Local source type specified for '{dataset_cfg.name}'. Please populate content at: {local_path.absolute()}")
                return
            raise ve

        try:
            downloaded_file = downloader.download(local_path)
            
            # Verify SHA256 integrity hash if configured and it is a single file
            if dataset_cfg.sha256 and downloaded_file.is_file():
                logger.info(f"Running integrity hash verification for '{dataset_cfg.name}'...")
                calculated_hash = self.calculate_sha256(downloaded_file)
                if calculated_hash.lower() != dataset_cfg.sha256.lower():
                    raise ValueError(
                        f"Integrity check failed for {downloaded_file}. Expected SHA256 '{dataset_cfg.sha256}', got '{calculated_hash}'"
                    )
                logger.info(f"Integrity check passed! Hash matched: {calculated_hash}")
            
            # Extract if downloaded_file is a compressed archive
            if downloaded_file.is_file() and downloaded_file.suffix.lower() in [".zip", ".tar", ".gz", ".tgz"]:
                ArchiveExtractor.extract(downloaded_file, local_path)
                
            logger.info(f"Dataset '{dataset_cfg.name}' successfully downloaded and prepared at {local_path.absolute()}")
        except Exception as e:
            logger.error(f"Download/Extraction process failed for dataset '{dataset_cfg.name}': {e}")
            raise e

    @staticmethod
    def calculate_sha256(file_path: Path) -> str:
        """Computes the SHA256 hash of a file on disk."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(65536), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def check_cache_valid(self) -> bool:
        """Checks if preprocessed cache config exists and matches current configs and files on disk."""
        if not self.metadata_csv_path.exists() or not self.cache_config_path.exists():
            return False
            
        try:
            # 1. Load cache config block
            with open(self.cache_config_path, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            
            # 2. Check preprocessing settings equivalence
            p_cfg = self.config.preprocessing
            cached_prep = cached_data.get("preprocessing", {})
            if (cached_prep.get("grayscale") != p_cfg.grayscale or
                cached_prep.get("resize_width") != p_cfg.resize.width or
                cached_prep.get("resize_height") != p_cfg.resize.height or
                cached_prep.get("resize_mode") != p_cfg.resize.mode or
                cached_prep.get("split_train") != p_cfg.split.train or
                cached_prep.get("split_seed") != p_cfg.split.seed):
                logger.info("Cache invalidated: Preprocessing configuration changed.")
                return False

            # 3. Check dataset versions matching
            cached_versions = cached_data.get("dataset_versions", {})
            for k, ds_cfg in self.config.datasets.items():
                if cached_versions.get(ds_cfg.name) != ds_cfg.version:
                    logger.info(f"Cache invalidated: Version changed for {ds_cfg.name}.")
                    return False

            # 4. Verify images exist on disk
            df = pd.read_csv(self.metadata_csv_path)
            if df.empty:
                return False
                
            base_dir = self.metadata_csv_path.parent.parent
            for _, row in df.iterrows():
                inp_rel = row.get("input_path")
                gt_rel = row.get("ground_truth_path")
                if not inp_rel or not gt_rel:
                    return False
                    
                inp_path = base_dir.parent / Path(inp_rel)
                gt_path = base_dir.parent / Path(gt_rel)
                if not inp_path.exists() or not gt_path.exists():
                    logger.warning(f"Cache file missing on disk: {inp_path} or {gt_path}")
                    return False
                    
            logger.info(f"Dataset cache verified (Fingerprint: {cached_data.get('fingerprint')}). Skipping preprocessing phase.")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to check cache validity: {e}. Skipping cache.")
            return False

    def prepare_dataset(self, dataset_cfg: DatasetSourceConfig) -> None:
        """Orchestrates detection, downloading, and verification for a single dataset."""
        available = self.is_dataset_locally_available(dataset_cfg)
        if available:
            logger.info(f"Dataset '{dataset_cfg.name}' detected locally at: {dataset_cfg.local_path}")
            return
        
        # Pull from remote
        self.download_and_extract_dataset(dataset_cfg)

    def generate_fingerprint_and_save(self) -> str:
        """Computes a SHA256-based fingerprint of the entire preprocessed dataset."""
        if not self.metadata_csv_path.exists():
            return ""

        logger.info("Computing SHA256-based dataset fingerprint...")
        try:
            df = pd.read_csv(self.metadata_csv_path)
            hashes = []
            base_dir = self.metadata_csv_path.parent.parent

            for idx, row in df.iterrows():
                inp_rel = row.get("input_path")
                if inp_rel:
                    inp_path = base_dir.parent / Path(inp_rel)
                    if inp_path.exists():
                        h = self.calculate_sha256(inp_path)
                        hashes.append((row.get("image_name", f"img_{idx}"), h))

            # Sort to guarantee order-reproducibility
            hashes.sort()
            fingerprint_input = "".join([h for name, h in hashes]).encode("utf-8")
            dataset_fingerprint = hashlib.sha256(fingerprint_input).hexdigest()

            # Save fingerprint in metadata
            df["fingerprint"] = dataset_fingerprint
            df.to_csv(self.metadata_csv_path, index=False)

            logger.info(f"Unified Dataset Fingerprint: {dataset_fingerprint}")
            return dataset_fingerprint
        except Exception as e:
            logger.error(f"Failed to calculate dataset fingerprint: {e}")
            return ""

    def save_cache_config(self, fingerprint: str) -> None:
        """Saves current configuration and versions metadata for caching checks."""
        p_cfg = self.config.preprocessing
        cache_data = {
            "fingerprint": fingerprint,
            "preprocessing": {
                "grayscale": p_cfg.grayscale,
                "resize_width": p_cfg.resize.width,
                "resize_height": p_cfg.resize.height,
                "resize_mode": p_cfg.resize.mode,
                "split_train": p_cfg.split.train,
                "split_seed": p_cfg.split.seed,
            },
            "dataset_versions": {ds_cfg.name: ds_cfg.version for k, ds_cfg in self.config.datasets.items()},
            "generated_at": datetime.datetime.now().isoformat()
        }
        with open(self.cache_config_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=4)
        logger.info(f"Saved cache config configuration to {self.cache_config_path}")

    def generate_dataset_card(self, stats: Dict[str, Any], validation: Dict[str, Any], fingerprint: str) -> None:
        """Generates the dataset_card.md documentation summarizing metadata metrics."""
        card_path = self.processed_dir / "dataset_card.md"
        logger.info(f"Generating dataset card at {card_path.absolute()}")

        p_cfg = self.config.preprocessing
        ds_versions_lines = "\n".join([f"- **{ds_cfg.name}**: version `{ds_cfg.version}`" for k, ds_cfg in self.config.datasets.items()])

        md_content = f"""# Dataset Card: Semiconductor Image Restoration Dataset

This dataset card documents structural specifications, statistics, preprocessing parameters, and validation logs for the semiconductor metrology restoration database.

## Dataset Overview
- **Total Images**: {stats.get("total_images", 0)} (Noisy / Ground Truth pairs)
- **Image Formats**: `{list(stats.get("formats", {}).keys())}`
- **Grayscale Dimensions**: {stats.get("dimensions", {}).get("avg_width", 0):.0f}×{stats.get("dimensions", {}).get("avg_height", 0):.0f} (H x W)
- **Dataset Fingerprint**: `{fingerprint}`
- **Validation Quality Score**: `{validation.get("quality_score", 0)}/100` (`{validation.get("status", "FAIL")}`)

## Configured Versions
{ds_versions_lines}

## Preprocessing Configuration
- **Grayscale Conversion**: `{p_cfg.grayscale}`
- **Resizing**: Width `{p_cfg.resize.width}`, Height `{p_cfg.resize.height}`, Mode `{p_cfg.resize.mode}`
- **Split Ratios**: Train `{p_cfg.split.train}`, Val `{p_cfg.split.val}`, Test `{p_cfg.split.test}`
- **Random Split Seed**: `{p_cfg.split.seed}`

## Metrology Statistics Summary
- **Noisy Inputs Mean / Std**: `{stats.get("noisy_stats", {}).get("mean", 0):.5f}` / `{stats.get("noisy_stats", {}).get("std", 0):.5f}`
- **Clean Targets Mean / Std**: `{stats.get("clean_stats", {}).get("mean", 0):.5f}` / `{stats.get("clean_stats", {}).get("std", 0):.5f}`

## Validation Audit Trail
- **Broken Files**: `{len(validation.get("broken_images", []))}`
- **Missing Files**: `{len(validation.get("missing_images", []))}`
- **Wrong Resolution Count**: `{len(validation.get("resolution_errors", []))}`
- **Duplicate Images**: `{len(validation.get("duplicate_images", []))}`
- **Filename Irregularities**: `{len(validation.get("filename_errors", []))}`

## Licensing
The raw data sources are derived from local experimental metrology databases,Zenodo public repository license standard Attribution-ShareAlike 4.0 International (CC BY-SA 4.0), and Hugging Face open restoration datasets. All processed derivatives are licensed under Creative Commons for Semiconductor Restoration Research.
"""
        with open(card_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        logger.info(f"Saved dataset card artifact to {card_path.absolute()}")

    def generate_experiment_template(self, fingerprint: str) -> None:
        """Generates the experiment.json metadata template for reproducible training runs."""
        template_path = self.processed_dir / "experiment.json"
        logger.info(f"Generating experiment template metadata at {template_path.absolute()}")

        template = {
            "experiment_name": "SemiconductorRestoration_Adaptive_Baseline",
            "timestamp": datetime.datetime.now().isoformat(),
            "reproducibility": {
                "dataset_fingerprint": fingerprint,
                "metadata_csv": str(self.metadata_csv_path.relative_to(self.processed_dir.parent.parent)),
                "dataset_versions": {ds_cfg.name: ds_cfg.version for k, ds_cfg in self.config.datasets.items()}
            },
            "training_configurations": {
                "epochs": 150,
                "batch_size": self.config.dataloader.batch_size,
                "learning_rate": 2.0e-4,
                "optimizer": {
                    "type": "AdamW",
                    "weight_decay": 1e-4,
                    "betas": [0.9, 0.99]
                },
                "lr_scheduler": {
                    "type": "CosineAnnealingLR",
                    "t_max": 150,
                    "eta_min": 1e-6
                },
                "loss": {
                    "type": "HybridLoss",
                    "weights": {
                        "l1": 0.5,
                        "ms_ssim": 0.5
                    }
                }
            },
            "model_zoo_settings": {
                "architecture": "NAFNet",
                "parameters": {
                    "width": 32,
                    "enc_blk_nums": [1, 1, 1, 28],
                    "dec_blk_nums": [1, 1, 1, 1]
                }
            }
        }

        with open(template_path, "w", encoding="utf-8") as f:
            json.dump(template, f, indent=4)
        logger.info(f"Saved experiment template to {template_path.absolute()}")

    def load(self, dataset_name: str, split: str = "train", is_training: bool = False) -> SemiconductorDataset:
        """Loads a preprocessed dataset subset directly into a SemiconductorDataset instance.

        Args:
            dataset_name (str): Name of the dataset (e.g. 'SEM', 'NIST').
            split (str): 'train', 'val', or 'test'.
            is_training (bool): Enforces augmentations settings if True.

        Returns:
            SemiconductorDataset: Instantiated PyTorch Dataset helper.
        """
        if not self.metadata_csv_path.exists():
            raise FileNotFoundError(f"Processed metadata file not found. Run prepare_and_process_all() first.")

        df = self.metadata_mgr.load_metadata()
        filtered = df[(df["dataset_name"].str.upper() == dataset_name.upper()) & (df["split"].str.lower() == split.lower())]
        if filtered.empty:
            raise ValueError(f"No records matched dataset '{dataset_name}' and split '{split}' in metadata registry.")

        base_dir = self.metadata_csv_path.parent.parent.parent
        return SemiconductorDataset(
            metadata_df=filtered,
            base_dir=base_dir,
            augmentation_opts=self.config.augmentation,
            is_training=is_training
        )

    def prepare_and_process_all(self, force_rebuild: bool = False) -> List[Dict[str, Any]]:
        """Orchestrates the entire global pipeline in a reorganized validation-first flow.

        Reorganization:
        1. Checks Cache.
        2. Prepares/Processes datasets if cache missing/invalid.
        3. STEP 1: VALIDATION runs immediately.
           - If status is FAIL, aborts the pipeline! No profiling, stats, or reports generated.
        4. STEP 2: STATISTICS GENERATOR.
        5. STEP 3: DATASET PROFILER. Updates metadata.csv and saves profile metrics.
        6. STEP 4: HTML REPORTER.
        7. STEP 5: Generate Dataset Card & Experiment template.
        """
        logger.info("Initializing Dataset Manager Orchestration...")

        is_cache_valid = self.check_cache_valid()
        if is_cache_valid and not force_rebuild:
            df = pd.read_csv(self.metadata_csv_path)
            records = df.to_dict(orient="records")
            logger.info("Loaded successfully from local processed cache.")
        else:
            if force_rebuild:
                logger.info("Force rebuild requested. Purging local processed files.")
            
            # 1. Pull dataset folders locally
            for key, ds_cfg in self.config.datasets.items():
                try:
                    self.prepare_dataset(ds_cfg)
                except Exception as e:
                    logger.error(
                        f"Skipping dataset configuration '{key}' due to download/initialization error: {e}"
                    )

            # 2. Preprocess raw content standardizing formats & sizing
            records = self.preprocessor.execute()

            if records:
                self.metadata_mgr.write_metadata(records)
                logger.info(f"Unified dataset metadata registry compiled at {self.metadata_csv_path}")
            else:
                logger.warning("No records preprocessed. Skipping metadata generation.")
                return []

        # Reorganized Pipeline Flow

        # STEP 1: VALIDATION
        logger.info("Pipeline Step 1: Executing dataset validation audit...")
        validator = DatasetValidator(
            self.metadata_csv_path,
            (self.config.preprocessing.resize.height, self.config.preprocessing.resize.width)
        )
        validation_report = validator.validate()
        validator.save_validation_report(self.processed_dir / "validation_report.json")

        if validation_report.get("status") == "FAIL":
            msg = (
                f"CRITICAL VALIDATION FAILURE. Pipeline aborted! Status: FAIL. "
                f"Validation score: {validation_report.get('quality_score')}/100. "
                f"Please fix broken/missing images before running other steps."
            )
            logger.error(msg)
            raise RuntimeError(msg)

        # Generate dataset fingerprints and update metadata.csv columns
        fingerprint = self.generate_fingerprint_and_save()
        self.save_cache_config(fingerprint)

        # STEP 2: STATISTICS GENERATOR
        logger.info("Pipeline Step 2: Generating numerical stats parameters...")
        stats_gen = DatasetStatsGenerator(self.metadata_csv_path)
        stats = stats_gen.generate_statistics()
        stats_gen.save_statistics(self.processed_dir / "dataset_statistics.json")

        # STEP 3: DATASET PROFILER (Updates metadata.csv)
        logger.info("Pipeline Step 3: Running dataset restoration profiler metadata task...")
        profiler = DatasetProfiler(self.metadata_csv_path)
        profiler_data = profiler.profile()
        profiler.save_profile(self.processed_dir / "dataset_profile.json")

        # Load updated records (with profile features)
        records_df = pd.read_csv(self.metadata_csv_path)
        records = records_df.to_dict(orient="records")

        # STEP 4: HTML REPORTER (Saves to self.reports_dir)
        logger.info(f"Pipeline Step 4: Compiling HTML Diagnostic report to {self.reports_dir}")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        reporter = DatasetReporter(self.metadata_csv_path)
        reporter.generate_html_report(
            stats=stats,
            validation=validation_report,
            config_datasets=self.config.datasets,
            output_html_path=self.reports_dir / "dataset_report.html"
        )

        # STEP 5: Generate Dataset Card and Experiment Template
        self.generate_dataset_card(stats, validation_report, fingerprint)
        self.generate_experiment_template(fingerprint)

        logger.info("Orchestration pipeline step execution complete.")
        return records
