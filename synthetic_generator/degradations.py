"""
Degradations module for the Semiconductor Image Restoration System (Module 5).
Implements a plugin registry pattern for degradation operators,
where every degradation is designed as an independent class that registers itself.
Uses a passed np.random.RandomState to ensure strict reproducibility and determinism.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List, Type
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class DegradationRegistry:
    """Plugin registry for registering and resolving degradation operators."""
    _registry: Dict[str, Type["BaseDegradation"]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a degradation subclass.

        Args:
            name (str): Unique key for the degradation.
        """
        def decorator(subclass: Type["BaseDegradation"]):
            cls._registry[name.lower()] = subclass
            return subclass
        return decorator

    @classmethod
    def get_operator(cls, name: str, **kwargs) -> "BaseDegradation":
        """Instantiates the registered degradation operator.

        Args:
            name (str): Key of the operator.
            **kwargs: Operator configuration arguments.

        Returns:
            BaseDegradation: The instantiated operator.
        """
        key = name.lower()
        if key not in cls._registry:
            raise ValueError(f"No degradation operator registered with key '{name}'")
        return cls._registry[key](**kwargs)


class BaseDegradation(ABC):
    """Abstract base class of all degradation operators."""

    @abstractmethod
    def apply(self, image: np.ndarray, state: np.random.RandomState) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Applies the degradation operator to the grayscale image.

        Args:
            image (np.ndarray): Grayscale input image, values in [0, 255] (uint8).
            state (np.random.RandomState): Deterministic random state.

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]:
                - Degraded image (uint8, same shape).
                - Dict containing the actual parameter values used in the application.
        """
        pass


@DegradationRegistry.register("gaussian_noise")
class GaussianNoise(BaseDegradation):
    """Adds additive Gaussian noise to the image."""

    def __init__(self, std_range: List[float] = None, mean: float = 0.0):
        self.std_range = std_range if std_range is not None else [0.01, 0.08]
        self.mean = mean

    def apply(self, image: np.ndarray, state: np.random.RandomState) -> Tuple[np.ndarray, Dict[str, Any]]:
        img_f = image.astype(np.float64) / 255.0
        std = state.uniform(self.std_range[0], self.std_range[1])
        noise = state.normal(self.mean, std, img_f.shape)
        out = np.clip(img_f + noise, 0.0, 1.0)
        return (out * 255.0).astype(np.uint8), {"noise_type": "gaussian", "std": float(std), "mean": float(self.mean)}


@DegradationRegistry.register("speckle_noise")
class SpeckleNoise(BaseDegradation):
    """Adds multiplicative Speckle noise to the image."""

    def __init__(self, std_range: List[float] = None):
        self.std_range = std_range if std_range is not None else [0.05, 0.2]

    def apply(self, image: np.ndarray, state: np.random.RandomState) -> Tuple[np.ndarray, Dict[str, Any]]:
        img_f = image.astype(np.float64) / 255.0
        std = state.uniform(self.std_range[0], self.std_range[1])
        noise = state.normal(0, std, img_f.shape)
        out = np.clip(img_f + img_f * noise, 0.0, 1.0)
        return (out * 255.0).astype(np.uint8), {"noise_type": "speckle", "std": float(std)}


@DegradationRegistry.register("poisson_noise")
class PoissonNoise(BaseDegradation):
    """Adds Poisson noise simulating electron shot noise."""

    def __init__(self, scale_range: List[float] = None):
        self.scale_range = scale_range if scale_range is not None else [10.0, 40.0]

    def apply(self, image: np.ndarray, state: np.random.RandomState) -> Tuple[np.ndarray, Dict[str, Any]]:
        img_f = image.astype(np.float64) / 255.0
        scale = state.uniform(self.scale_range[0], self.scale_range[1])
        # Electron count is proportional to scale
        scaled = np.clip(img_f * scale, 1e-4, None)
        noisy = state.poisson(scaled) / scale
        out = np.clip(noisy, 0.0, 1.0)
        return (out * 255.0).astype(np.uint8), {"noise_type": "poisson", "scale": float(scale)}


@DegradationRegistry.register("salt_pepper_noise")
class SaltPepperNoise(BaseDegradation):
    """Adds Salt & Pepper noise to the image."""

    def __init__(self, amount_range: List[float] = None, salt_vs_pepper: float = 0.5):
        self.amount_range = amount_range if amount_range is not None else [0.005, 0.03]
        self.salt_vs_pepper = salt_vs_pepper

    def apply(self, image: np.ndarray, state: np.random.RandomState) -> Tuple[np.ndarray, Dict[str, Any]]:
        out = image.copy()
        amount = state.uniform(self.amount_range[0], self.amount_range[1])
        
        # Salt
        num_salt = np.ceil(amount * image.size * self.salt_vs_pepper).astype(int)
        coords = [state.randint(0, i - 1, num_salt) for i in image.shape]
        out[tuple(coords)] = 255

        # Pepper
        num_pepper = np.ceil(amount * image.size * (1.0 - self.salt_vs_pepper)).astype(int)
        coords = [state.randint(0, i - 1, num_pepper) for i in image.shape]
        out[tuple(coords)] = 0

        return out, {"noise_type": "salt_pepper", "amount": float(amount)}


@DegradationRegistry.register("sensor_noise")
class SensorNoise(BaseDegradation):
    """Combines Poisson (shot) noise and Gaussian (electronic) read noise."""

    def __init__(self, poisson_scale_range: List[float] = None, gaussian_std_range: List[float] = None):
        self.poisson_scale_range = poisson_scale_range if poisson_scale_range is not None else [15.0, 50.0]
        self.gaussian_std_range = gaussian_std_range if gaussian_std_range is not None else [0.01, 0.05]

    def apply(self, image: np.ndarray, state: np.random.RandomState) -> Tuple[np.ndarray, Dict[str, Any]]:
        img_f = image.astype(np.float64) / 255.0
        
        poisson_scale = state.uniform(self.poisson_scale_range[0], self.poisson_scale_range[1])
        gaussian_std = state.uniform(self.gaussian_std_range[0], self.gaussian_std_range[1])

        # 1. Poisson shot noise
        scaled = np.clip(img_f * poisson_scale, 1e-4, None)
        poisson_noisy = state.poisson(scaled) / poisson_scale

        # 2. Gaussian readout noise
        noise_g = state.normal(0, gaussian_std, img_f.shape)
        out = np.clip(poisson_noisy + noise_g, 0.0, 1.0)
        
        return (out * 255.0).astype(np.uint8), {
            "noise_type": "sensor",
            "poisson_scale": float(poisson_scale),
            "gaussian_std": float(gaussian_std)
        }


@DegradationRegistry.register("gaussian_blur")
class GaussianBlur(BaseDegradation):
    """Applies Gaussian Blur to the image."""

    def __init__(self, kernel_range: List[int] = None, sigma_range: List[float] = None):
        self.kernel_range = kernel_range if kernel_range is not None else [3, 11]
        self.sigma_range = sigma_range if sigma_range is not None else [0.5, 3.0]

    def apply(self, image: np.ndarray, state: np.random.RandomState) -> Tuple[np.ndarray, Dict[str, Any]]:
        k_min, k_max = self.kernel_range
        # Kernels must be odd
        odd_kernels = [k for k in range(k_min, k_max + 1) if k % 2 != 0]
        if not odd_kernels:
            odd_kernels = [3]
        ksize = int(state.choice(odd_kernels))
        sigma = state.uniform(self.sigma_range[0], self.sigma_range[1])
        
        out = cv2.GaussianBlur(image, (ksize, ksize), sigma)
        return out, {"blur_type": "gaussian", "kernel_size": ksize, "sigma": float(sigma)}


@DegradationRegistry.register("defocus_blur")
class DefocusBlur(BaseDegradation):
    """Applies circular Defocus Blur to simulate lens out-of-focus states."""

    def __init__(self, radius_range: List[int] = None):
        self.radius_range = radius_range if radius_range is not None else [2, 8]

    def apply(self, image: np.ndarray, state: np.random.RandomState) -> Tuple[np.ndarray, Dict[str, Any]]:
        radius = int(state.randint(self.radius_range[0], self.radius_range[1] + 1))
        
        # Create disc kernel
        kernel_size = 2 * radius + 1
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        cv2.circle(kernel, (radius, radius), radius, 1.0, -1)
        kernel = kernel / np.sum(kernel)

        out = cv2.filter2D(image, -1, kernel)
        return out, {"blur_type": "defocus", "radius": radius, "kernel_size": kernel_size}


@DegradationRegistry.register("motion_blur")
class MotionBlur(BaseDegradation):
    """Applies linear Motion Blur representing spatial shifts."""

    def __init__(self, size_range: List[int] = None, angle_range: List[float] = None):
        self.size_range = size_range if size_range is not None else [5, 21]
        self.angle_range = angle_range if angle_range is not None else [0.0, 360.0]

    def apply(self, image: np.ndarray, state: np.random.RandomState) -> Tuple[np.ndarray, Dict[str, Any]]:
        size = int(state.randint(self.size_range[0], self.size_range[1] + 1))
        angle = state.uniform(self.angle_range[0], self.angle_range[1])

        # Generate directional motion kernel
        kernel = np.zeros((size, size), dtype=np.float32)
        # Find coordinates along the line segment
        cx, cy = size // 2, size // 2
        rad = np.deg2rad(angle)
        dx = np.cos(rad) * cx
        dy = np.sin(rad) * cy
        
        cv2.line(kernel, (int(cx - dx), int(cy - dy)), (int(cx + dx), int(cy + dy)), 1.0, thickness=1)
        kernel = kernel / np.sum(kernel)

        out = cv2.filter2D(image, -1, kernel)
        return out, {"blur_type": "motion", "kernel_size": size, "angle": float(angle)}


@DegradationRegistry.register("downsampling")
class Downsampling(BaseDegradation):
    """Resizes the image to a lower resolution."""

    def __init__(self, scale_range: List[float] = None):
        self.scale_range = scale_range if scale_range is not None else [0.5, 0.9]

    def apply(self, image: np.ndarray, state: np.random.RandomState) -> Tuple[np.ndarray, Dict[str, Any]]:
        h, w = image.shape[:2]
        scale = state.uniform(self.scale_range[0], self.scale_range[1])
        w_new, h_new = int(w * scale), int(h * scale)
        
        # Enforce minimum size of 16
        w_new = max(16, w_new)
        h_new = max(16, h_new)

        interp = int(state.choice([cv2.INTER_AREA, cv2.INTER_LINEAR]))
        out = cv2.resize(image, (w_new, h_new), interpolation=interp)
        
        return out, {"resolution_loss": float(1.0 - scale), "scale": float(scale)}


@DegradationRegistry.register("bicubic_degradation")
class BicubicDegradation(BaseDegradation):
    """Downsamples image and scales it back up using bicubic interpolation."""

    def __init__(self, scale_range: List[float] = None):
        self.scale_range = scale_range if scale_range is not None else [0.25, 0.75]

    def apply(self, image: np.ndarray, state: np.random.RandomState) -> Tuple[np.ndarray, Dict[str, Any]]:
        h, w = image.shape[:2]
        scale = state.uniform(self.scale_range[0], self.scale_range[1])
        w_sub, h_sub = max(8, int(w * scale)), max(8, int(h * scale))

        # Downsample
        down = cv2.resize(image, (w_sub, h_sub), interpolation=cv2.INTER_AREA)
        # Upsample back using bicubic interpolation
        out = cv2.resize(down, (w, h), interpolation=cv2.INTER_CUBIC)

        return out, {"blur_type": "bicubic", "scale": float(scale), "resolution_loss": float(1.0 - scale)}


@DegradationRegistry.register("jpeg_compression")
class JPEGCompression(BaseDegradation):
    """Applies JPEG compression artifacts to the image."""

    def __init__(self, quality_range: List[int] = None):
        self.quality_range = quality_range if quality_range is not None else [20, 85]

    def apply(self, image: np.ndarray, state: np.random.RandomState) -> Tuple[np.ndarray, Dict[str, Any]]:
        quality = int(state.randint(self.quality_range[0], self.quality_range[1] + 1))
        
        # Compress to JPEG in memory
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        result, enc_img = cv2.imencode('.jpg', image, encode_param)
        
        # Decompress back
        out = cv2.imdecode(enc_img, cv2.IMREAD_GRAYSCALE)
        
        return out, {"compression_quality": quality}


@DegradationRegistry.register("brightness_variation")
class BrightnessVariation(BaseDegradation):
    """Varies the image brightness (offset shift)."""

    def __init__(self, delta_range: List[int] = None):
        self.delta_range = delta_range if delta_range is not None else [-40, 40]

    def apply(self, image: np.ndarray, state: np.random.RandomState) -> Tuple[np.ndarray, Dict[str, Any]]:
        delta = int(state.randint(self.delta_range[0], self.delta_range[1] + 1))
        out = np.clip(image.astype(np.int16) + delta, 0, 255).astype(np.uint8)
        return out, {"brightness_factor": float(delta)}


@DegradationRegistry.register("contrast_variation")
class ContrastVariation(BaseDegradation):
    """Varies the image contrast."""

    def __init__(self, alpha_range: List[float] = None):
        self.alpha_range = alpha_range if alpha_range is not None else [0.6, 1.6]

    def apply(self, image: np.ndarray, state: np.random.RandomState) -> Tuple[np.ndarray, Dict[str, Any]]:
        alpha = state.uniform(self.alpha_range[0], self.alpha_range[1])
        img_f = image.astype(np.float64)
        mean = np.mean(img_f)
        out = np.clip((img_f - mean) * alpha + mean, 0, 255).astype(np.uint8)
        return out, {"contrast_factor": float(alpha)}


@DegradationRegistry.register("gamma_correction")
class GammaCorrection(BaseDegradation):
    """Applies non-linear Gamma Correction to the image."""

    def __init__(self, gamma_range: List[float] = None):
        self.gamma_range = gamma_range if gamma_range is not None else [0.5, 1.8]

    def apply(self, image: np.ndarray, state: np.random.RandomState) -> Tuple[np.ndarray, Dict[str, Any]]:
        gamma = state.uniform(self.gamma_range[0], self.gamma_range[1])
        
        # Build lookup table
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255.0 for i in np.arange(0, 256)]).astype("uint8")
        
        out = cv2.LUT(image, table)
        return out, {"gamma_factor": float(gamma)}


@DegradationRegistry.register("scanning_charging_streaks")
class ScanningChargingStreaks(BaseDegradation):
    """Applies horizontal raster charging streaks simulating insulative SEM charging glow."""

    def __init__(self, streaks_range: List[int] = None, strength_range: List[float] = None):
        self.streaks_range = streaks_range if streaks_range is not None else [1, 4]
        self.strength_range = strength_range if strength_range is not None else [0.1, 0.35]

    def apply(self, image: np.ndarray, state: np.random.RandomState) -> Tuple[np.ndarray, Dict[str, Any]]:
        h, w = image.shape[:2]
        img_f = image.astype(np.float64) / 255.0
        
        num_streaks = int(state.randint(self.streaks_range[0], self.streaks_range[1] + 1))
        strength_used = 0.0

        for _ in range(num_streaks):
            y = int(state.randint(0, h))
            strength = state.uniform(self.strength_range[0], self.strength_range[1])
            strength_used = max(strength_used, strength)
            
            glow_height = int(state.randint(2, 7))
            for offset in range(-glow_height, glow_height):
                if 0 <= y + offset < h:
                    falloff = 1.0 - (abs(offset) / glow_height)
                    img_f[y + offset, :] += strength * falloff

        out = np.clip(img_f, 0.0, 1.0)
        return (out * 255.0).astype(np.uint8), {"charging_artifacts": True, "streaks_count": num_streaks, "max_streak_strength": float(strength_used)}
