"""
Configuration management module for the Semiconductor Image Restoration Dataset System.
Loads, validates, and stores settings from a YAML configuration file.
"""

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml

logger = logging.getLogger(__name__)

@dataclass
class PathsConfig:
    """Paths workspace configuration."""
    raw_dir: Path
    processed_dir: Path
    reports_dir: Path
    log_file: Path

@dataclass
class ResizeConfig:
    """Resize parameters configuration."""
    enabled: bool
    width: int
    height: int
    mode: str
    interpolation: str

@dataclass
class SplitConfig:
    """Dataset splitting configurations."""
    train: float
    val: float
    test: float
    seed: int

@dataclass
class PreprocessingConfig:
    """Preprocessing parameters configuration."""
    grayscale: bool
    resize: ResizeConfig
    split: SplitConfig

@dataclass
class AugmentationConfig:
    """Synced dataset augmentation details."""
    enabled: bool
    horizontal_flip: bool
    vertical_flip: bool
    random_rotation: bool
    rotation_degrees: int
    random_crop: bool
    crop_size: List[int]

@dataclass
class DataLoaderConfig:
    """PyTorch DataLoader configuration parameters."""
    batch_size: int
    num_workers: int
    shuffle_train: bool
    pin_memory: bool
    drop_last: bool

@dataclass
class SyntheticConfig:
    """Synthetic SEM noise simulator parameters."""
    image_size: List[int]
    noise_type: str
    poisson_scale: float
    gaussian_sigma: float
    charging_artifacts: bool
    beam_blur: bool
    blur_sigma: float

@dataclass
class DatasetSourceConfig:
    """Configuration for a specific dataset source and its remote downloading settings."""
    name: str
    local_path: Path
    source_type: str
    record_id: Optional[str] = field(default=None)
    file_name: Optional[str] = field(default=None)
    repo_id: Optional[str] = field(default=None)
    file_id: Optional[str] = field(default=None)
    dataset_slug: Optional[str] = field(default=None)
    version: str = field(default="1.0")
    sha256: Optional[str] = field(default=None)

@dataclass
class SyntheticGeneratorConfig:
    """Synthetic Dataset Generator configuration details."""
    enabled: bool
    num_samples: int
    clean_source_dir: Path
    output_dir: Path
    seed: int
    preset: str
    multiprocessing: bool
    num_workers: int
    pipeline: List[Dict[str, Any]]

@dataclass
class PipelineConfig:
    """Global configuration structure."""
    paths: PathsConfig
    preprocessing: PreprocessingConfig
    augmentation: AugmentationConfig
    dataloader: DataLoaderConfig
    synthetic: SyntheticConfig
    synthetic_generator: SyntheticGeneratorConfig
    datasets: Dict[str, DatasetSourceConfig] = field(default_factory=dict)


    @classmethod
    def load_from_yaml(cls, yaml_path: Path) -> "PipelineConfig":
        """Loads configuration from a YAML file, runs validation, and returns config dataclass instance.

        Args:
            yaml_path (Path): Path to yaml config file.

        Returns:
            PipelineConfig: The instantiated configurations.

        Raises:
            FileNotFoundError: If configuration file is missing.
            ValueError: If configuration values fail standard checks.
        """
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        try:
            with open(yaml_path, "r") as f:
                data: Dict[str, Any] = yaml.safe_load(f)
        except Exception as e:
            raise ValueError(f"Failed to parse configuration YAML: {e}")

        # Config extraction and mapping
        try:
            paths_data = data.get("paths", {})
            paths = PathsConfig(
                raw_dir=Path(paths_data.get("raw_dir", "./datasets")),
                processed_dir=Path(paths_data.get("processed_dir", "./datasets/processed")),
                reports_dir=Path(paths_data.get("reports_dir", "./reports")),
                log_file=Path(paths_data.get("log_file", "./dataset_pipeline.log")),
            )

            prep_data = data.get("preprocessing", {})
            resize_data = prep_data.get("resize", {})
            resize = ResizeConfig(
                enabled=bool(resize_data.get("enabled", True)),
                width=int(resize_data.get("width", 512)),
                height=int(resize_data.get("height", 512)),
                mode=str(resize_data.get("mode", "pad")),
                interpolation=str(resize_data.get("interpolation", "lanczos")),
            )

            split_data = prep_data.get("split", {})
            split = SplitConfig(
                train=float(split_data.get("train", 0.8)),
                val=float(split_data.get("val", 0.1)),
                test=float(split_data.get("test", 0.1)),
                seed=int(split_data.get("seed", 42)),
            )

            preprocessing = PreprocessingConfig(
                grayscale=bool(prep_data.get("grayscale", True)),
                resize=resize,
                split=split,
            )

            aug_data = data.get("augmentation", {})
            augmentation = AugmentationConfig(
                enabled=bool(aug_data.get("enabled", True)),
                horizontal_flip=bool(aug_data.get("horizontal_flip", True)),
                vertical_flip=bool(aug_data.get("vertical_flip", True)),
                random_rotation=bool(aug_data.get("random_rotation", True)),
                rotation_degrees=int(aug_data.get("rotation_degrees", 90)),
                random_crop=bool(aug_data.get("random_crop", False)),
                crop_size=list(aug_data.get("crop_size", [256, 256])),
            )

            dl_data = data.get("dataloader", {})
            dataloader = DataLoaderConfig(
                batch_size=int(dl_data.get("batch_size", 8)),
                num_workers=int(dl_data.get("num_workers", 2)),
                shuffle_train=bool(dl_data.get("shuffle_train", True)),
                pin_memory=bool(dl_data.get("pin_memory", True)),
                drop_last=bool(dl_data.get("drop_last", False)),
            )

            synth_data = data.get("synthetic", {})
            synthetic = SyntheticConfig(
                image_size=list(synth_data.get("image_size", [1024, 1024])),
                noise_type=str(synth_data.get("noise_type", "poisson_gaussian")),
                poisson_scale=float(synth_data.get("poisson_scale", 15.0)),
                gaussian_sigma=float(synth_data.get("gaussian_sigma", 0.04)),
                charging_artifacts=bool(synth_data.get("charging_artifacts", True)),
                beam_blur=bool(synth_data.get("beam_blur", True)),
                blur_sigma=float(synth_data.get("blur_sigma", 1.2)),
            )

            datasets_data = data.get("datasets", {})
            datasets = {}
            for k, v in datasets_data.items():
                datasets[k] = DatasetSourceConfig(
                    name=str(v.get("name", k)),
                    local_path=Path(v.get("local_path", f"./datasets/{k}")),
                    source_type=str(v.get("source_type", "local")),
                    record_id=v.get("record_id"),
                    file_name=v.get("file_name"),
                    repo_id=v.get("repo_id"),
                    file_id=v.get("file_id"),
                    dataset_slug=v.get("dataset_slug"),
                    version=str(v.get("version", "1.0")),
                    sha256=v.get("sha256"),
                )

            sg_data = data.get("synthetic_generator", {})
            synthetic_generator = SyntheticGeneratorConfig(
                enabled=bool(sg_data.get("enabled", False)),
                num_samples=int(sg_data.get("num_samples", 50)),
                clean_source_dir=Path(sg_data.get("clean_source_dir", "./datasets/clean_sources")),
                output_dir=Path(sg_data.get("output_dir", "./datasets/Synthetic_Generated")),
                seed=int(sg_data.get("seed", 42)),
                preset=str(sg_data.get("preset", "sem")),
                multiprocessing=bool(sg_data.get("multiprocessing", True)),
                num_workers=int(sg_data.get("num_workers", 4)),
                pipeline=list(sg_data.get("pipeline", [])),
            )

            config = cls(
                paths=paths,
                preprocessing=preprocessing,
                augmentation=augmentation,
                dataloader=dataloader,
                synthetic=synthetic,
                synthetic_generator=synthetic_generator,
                datasets=datasets,
            )
            config.validate()
            return config


        except KeyError as ke:
            raise ValueError(f"Missing required configuration key: {ke}")
        except TypeError as te:
            raise ValueError(f"Configuration value type mismatch: {te}")

    def validate(self) -> None:
        """Validates configuration parameters.

        Raises:
            ValueError: If parameters are invalid.
        """
        # Validate splits
        s = self.preprocessing.split
        if not (0.0 <= s.train <= 1.0) or not (0.0 <= s.val <= 1.0) or not (0.0 <= s.test <= 1.0):
            raise ValueError("Split ratios must be between 0.0 and 1.0.")
        if not abs(s.train + s.val + s.test - 1.0) < 1e-9:
            raise ValueError(
                f"Split ratios must sum to 1.0. Current sum: {s.train + s.val + s.test}"
            )

        # Validate dimensions
        r = self.preprocessing.resize
        if r.enabled:
            if r.width <= 0 or r.height <= 0:
                raise ValueError("Resize dimensions must be positive integers.")
            if r.mode not in ["pad", "crop", "stretch"]:
                raise ValueError(
                    f"Resize mode '{r.mode}' is invalid. Options: 'pad', 'crop', 'stretch'."
                )
            if r.interpolation not in ["lanczos", "area", "cubic", "linear"]:
                raise ValueError(
                    f"Interpolation mode '{r.interpolation}' is invalid."
                )

        # Validate loader
        dl = self.dataloader
        if dl.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        if dl.num_workers < 0:
            raise ValueError("num_workers must be non-negative.")
