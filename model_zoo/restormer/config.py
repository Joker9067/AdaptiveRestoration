"""
Configuration mapping details for restormer.
"""
from typing import Dict, Any

DEFAULT_CONFIG: Dict[str, Any] = {
    "epochs": 5,
    "batch_size": 4,
    "learning_rate": 1e-3,
    "optimizer": "Adam",
    "lr_scheduler": "CosineAnnealingLR",
    "early_stopping_patience": 5,
    "grad_clip": 5.0,
    "amp": True,
    "multi_gpu": False,
    "loss_weights": {
        "l1": 1.0,
        "ssim": 0.2
    },
    "checkpoint_dir": "./checkpoints",
    "tensorboard_dir": "./runs"
}

def get_config() -> Dict[str, Any]:
    return DEFAULT_CONFIG.copy()
