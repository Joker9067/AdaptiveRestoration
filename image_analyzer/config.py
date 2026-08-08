"""
Configuration module for the Physics-Guided Image Analyzer (Module 7).
"""

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List


@dataclass
class HyperparametersConfig:
    batch_size: int = 4
    learning_rate: float = 1e-3
    epochs: int = 3
    early_stopping_patience: int = 3
    amp: bool = True
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        "noise_type": 1.0,
        "blur_type": 1.0,
        "severity": 1.0,
        "noise_level": 1.0,
        "blur_strength": 1.0,
        "resolution_loss": 1.0,
        "compression_quality": 1.0,
        "brightness": 1.0,
        "contrast": 1.0,
        "gamma": 1.0,
        "edge_density": 1.0,
        "texture_complexity": 1.0,
        "entropy": 1.0,
        "confidence": 1.0
    })


@dataclass
class AnalyzerConfig:
    model_name: str = "efficientnet_b0"
    hyperparameters: HyperparametersConfig = field(default_factory=HyperparametersConfig)
    checkpoint_dir: str = "./checkpoints/analyzer"
    tensorboard_dir: str = "./runs/analyzer"
    reports_dir: str = "./reports"

    @classmethod
    def load_from_yaml(cls, yaml_path: Path) -> "AnalyzerConfig":
        if not yaml_path.exists():
            return cls()
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        
        analyzer_data = data.get("image_analyzer", {})
        hparams_data = analyzer_data.get("hyperparameters", {})
        
        hparams = HyperparametersConfig(
            batch_size=hparams_data.get("batch_size", 4),
            learning_rate=hparams_data.get("learning_rate", 1e-3),
            epochs=hparams_data.get("epochs", 3),
            early_stopping_patience=hparams_data.get("early_stopping_patience", 3),
            amp=hparams_data.get("amp", True),
            loss_weights=hparams_data.get("loss_weights", cls().hyperparameters.loss_weights)
        )
        
        return cls(
            model_name=analyzer_data.get("model_name", "efficientnet_b0"),
            hyperparameters=hparams,
            checkpoint_dir=analyzer_data.get("checkpoint_dir", "./checkpoints/analyzer"),
            tensorboard_dir=analyzer_data.get("tensorboard_dir", "./runs/analyzer"),
            reports_dir=analyzer_data.get("reports_dir", "./reports")
        )
