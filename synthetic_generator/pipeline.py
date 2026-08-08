"""
Degradation Pipeline orchestrator module for the Semiconductor Image Restoration System (Module 5).
Combines registered degradation operators, manages preset configurations,
applies operators sequentially and deterministically, and calculates
overall degradation severity scores.
"""

import logging
from typing import Dict, Any, List, Tuple
import numpy as np

from synthetic_generator.degradations import DegradationRegistry, BaseDegradation

logger = logging.getLogger(__name__)

# Preset configuration dictionaries mapping degradation keys to parameters and probabilities
PRESETS: Dict[str, List[Dict[str, Any]]] = {
    "sem": [
        {
            "name": "gaussian_blur",
            "probability": 0.8,
            "params": {"kernel_range": [3, 9], "sigma_range": [0.5, 1.8]}
        },
        {
            "name": "scanning_charging_streaks",
            "probability": 0.7,
            "params": {"streaks_range": [1, 3], "strength_range": [0.08, 0.25]}
        },
        {
            "name": "sensor_noise",
            "probability": 1.0,
            "params": {"poisson_scale_range": [10.0, 35.0], "gaussian_std_range": [0.01, 0.045]}
        },
        {
            "name": "contrast_variation",
            "probability": 0.5,
            "params": {"alpha_range": [0.75, 1.35]}
        }
    ],
    "electron_microscopy": [
        {
            "name": "defocus_blur",
            "probability": 0.6,
            "params": {"radius_range": [2, 5]}
        },
        {
            "name": "gaussian_noise",
            "probability": 0.9,
            "params": {"std_range": [0.02, 0.08]}
        },
        {
            "name": "gamma_correction",
            "probability": 0.5,
            "params": {"gamma_range": [0.7, 1.4]}
        }
    ],
    "general_microscopy": [
        {
            "name": "gaussian_blur",
            "probability": 0.7,
            "params": {"kernel_range": [3, 11], "sigma_range": [0.5, 2.0]}
        },
        {
            "name": "speckle_noise",
            "probability": 0.8,
            "params": {"std_range": [0.05, 0.15]}
        },
        {
            "name": "brightness_variation",
            "probability": 0.5,
            "params": {"delta_range": [-25, 25]}
        }
    ]
}


class DegradationPipeline:
    """Manages composing and running sequential degradation operators on clean images."""

    def __init__(self, preset_name: str = "sem", custom_pipeline: List[Dict[str, Any]] = None):
        """Initializes the degradation pipeline.

        Args:
            preset_name (str): Selector matching 'sem', 'electron_microscopy', 'general_microscopy', or 'custom'.
            custom_pipeline (List[Dict[str, Any]]): Custom pipeline configurations.
        """
        self.preset_name = preset_name.lower()
        self.operators: List[Tuple[str, float, BaseDegradation]] = []
        self._initialize_pipeline(custom_pipeline)

    def _initialize_pipeline(self, custom_pipeline: List[Dict[str, Any]] = None) -> None:
        """Instantiates degradation operators based on resolved configuration."""
        configs = []
        if self.preset_name == "custom":
            if custom_pipeline:
                configs = custom_pipeline
            else:
                logger.warning("Custom preset selected but no custom_pipeline config provided. Defaulting to empty pipeline.")
        else:
            if self.preset_name in PRESETS:
                configs = PRESETS[self.preset_name]
                logger.info(f"Loading preset configuration mapping for '{self.preset_name}'")
            else:
                logger.warning(f"Unknown preset '{self.preset_name}'. Fallback to empty pipeline.")

        for op_cfg in configs:
            name = op_cfg.get("name")
            prob = op_cfg.get("probability", 1.0)
            params = op_cfg.get("params", {})
            try:
                operator = DegradationRegistry.get_operator(name, **params)
                self.operators.append((name, prob, operator))
                logger.debug(f"Loaded pipeline operator '{name}' with probability {prob}")
            except Exception as e:
                logger.error(f"Failed to initialize degradation operator '{name}': {e}")

    @staticmethod
    def calculate_severity(history: Dict[str, Any]) -> Tuple[float, str]:
        """Calculates degradation severity score and maps to a discrete level.

        Args:
            history (Dict[str, Any]): Dictionary of applied degradation parameters.

        Returns:
            Tuple[float, str]: (Severity Score [0.0, 1.0], Severity Level name)
        """
        score_acc = 0.0
        count = 0
        
        for op_name, params in history.items():
            if "noise" in op_name:
                if "std" in params:
                    score_acc += min(1.0, params["std"] / 0.12)
                elif "scale" in params:
                    score_acc += min(1.0, 20.0 / params["scale"])
                elif "poisson_scale" in params:
                    p_score = min(1.0, 20.0 / params["poisson_scale"])
                    g_score = min(1.0, params.get("gaussian_std", 0.0) / 0.06)
                    score_acc += (p_score + g_score) / 2.0
                elif "amount" in params:
                    score_acc += min(1.0, params["amount"] / 0.04)
                count += 1
                
            elif "blur" in op_name or "bicubic" in op_name:
                if "sigma" in params:
                    score_acc += min(1.0, params["sigma"] / 2.5)
                elif "radius" in params:
                    score_acc += min(1.0, params["radius"] / 8.0)
                elif "kernel_size" in params:
                    score_acc += min(1.0, params["kernel_size"] / 15.0)
                elif "scale" in params:
                    score_acc += min(1.0, (1.0 - params["scale"]) / 0.7)
                count += 1
                
            elif "downsampling" in op_name:
                score_acc += min(1.0, (1.0 - params.get("scale", 1.0)) / 0.6)
                count += 1
                
            elif "jpeg" in op_name:
                score_acc += min(1.0, (100 - params["compression_quality"]) / 80.0)
                count += 1
                
            elif "streaks" in op_name:
                streaks = params.get("streaks_count", 1)
                strength = params.get("max_streak_strength", 0.1)
                score_acc += min(1.0, (streaks * strength) / 1.0)
                count += 1

            elif "brightness" in op_name:
                score_acc += min(0.3, abs(params.get("brightness_factor", 0.0)) / 40.0)
                count += 1

            elif "contrast" in op_name:
                factor = params.get("contrast_factor", 1.0)
                score_acc += min(0.3, abs(1.0 - factor) / 0.5)
                count += 1

        if count > 0:
            severity_score = min(1.0, score_acc / count)
        else:
            severity_score = 0.0

        # Map to descriptive string labels
        if severity_score <= 0.25:
            level = "Easy"
        elif severity_score <= 0.50:
            level = "Medium"
        elif severity_score <= 0.75:
            level = "Hard"
        else:
            level = "Extreme"

        return float(severity_score), level

    def run(self, image: np.ndarray, state: np.random.RandomState) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Sequentially applies registered degradations to the clean image.

        Args:
            image (np.ndarray): Clean input image (uint8).
            state (np.random.RandomState): Random state for determinism.

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]:
                - Degraded image (uint8).
                - Dictionary containing the pipeline execution parameters and history metrics.
        """
        out = image.copy()
        applied_history = {}
        active_sequence = []

        for name, prob, operator in self.operators:
            # Check probability check
            if state.uniform(0.0, 1.0) <= prob:
                out, op_params = operator.apply(out, state)
                applied_history[name] = op_params
                active_sequence.append(name)

        # Calculate final severity score
        severity_score, severity_level = self.calculate_severity(applied_history)

        metadata = {
            "applied_sequence": active_sequence,
            "parameters": applied_history,
            "severity_score": severity_score,
            "severity_level": severity_level
        }

        return out, metadata
