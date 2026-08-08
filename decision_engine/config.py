"""
Configuration module for the Adaptive Decision Engine (Module 8).
"""

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any


@dataclass
class DecisionHyperparametersConfig:
    batch_size: int = 4
    learning_rate: float = 1e-3
    epochs: int = 3
    early_stopping_patience: int = 3
    amp: bool = True
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        "reconstruction": 1.0,
        "ssim": 0.5,
        "entropy_reg": 0.1,  # Keeps expert weights from collapsing to single expert
        "order_cls": 1.0     # Supervision for optimal sequence ordering prediction
    })


@dataclass
class DecisionEngineConfig:
    hyperparameters: DecisionHyperparametersConfig = field(default_factory=DecisionHyperparametersConfig)
    checkpoint_dir: str = "./checkpoints/decision_engine"
    tensorboard_dir: str = "./runs/decision_engine"
    reports_dir: str = "./reports"

    @classmethod
    def load_from_yaml(cls, yaml_path: Path) -> "DecisionEngineConfig":
        if not yaml_path.exists():
            return cls()
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            
        de_data = data.get("decision_engine", {})
        hparams_data = de_data.get("hyperparameters", {})
        
        hparams = DecisionHyperparametersConfig(
            batch_size=hparams_data.get("batch_size", 4),
            learning_rate=hparams_data.get("learning_rate", 1e-3),
            epochs=hparams_data.get("epochs", 3),
            early_stopping_patience=hparams_data.get("early_stopping_patience", 3),
            amp=hparams_data.get("amp", True),
            loss_weights=hparams_data.get("loss_weights", cls().hyperparameters.loss_weights)
        )
        
        return cls(
            hyperparameters=hparams,
            checkpoint_dir=de_data.get("checkpoint_dir", "./checkpoints/decision_engine"),
            tensorboard_dir=de_data.get("tensorboard_dir", "./runs/decision_engine"),
            reports_dir=de_data.get("reports_dir", "./reports")
        )
