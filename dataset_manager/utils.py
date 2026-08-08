"""
Utility module for the Semiconductor Image Restoration System.
Provides:
1. Signal Processing Metrics: MSE, PSNR, SSIM, SNR (Local & Global).
2. Code logging setup helper.
"""

import logging
import sys
from pathlib import Path
from typing import Tuple, Union
import cv2
import numpy as np

logger = logging.getLogger(__name__)

def setup_pipeline_logging(
    log_file: Union[str, Path] = "./dataset_pipeline.log",
    level: int = logging.INFO,
) -> None:
    """Configures system-wide logging to output to both console and a log file.

    Args:
        log_file (Union[str, Path]): Path to log file destination.
        level (int): Logging severity thresholds (e.g. logging.INFO).
    """
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    handlers = [
        logging.FileHandler(log_path, mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]

    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )
    logger.info(f"Logging initialized. Writing logs to: {log_path.absolute()}")


def calculate_mse(clean: np.ndarray, noisy: np.ndarray) -> float:
    """Computes Mean Squared Error (MSE) between two flat 2D images.

    Args:
        clean (np.ndarray): Ground truth target image (uint8 or float).
        noisy (np.ndarray): Noisy input image (uint8 or float).

    Returns:
        float: MSE score.
    """
    c_arr = clean.astype(np.float64)
    n_arr = noisy.astype(np.float64)
    return float(np.mean((c_arr - n_arr) ** 2))


def calculate_psnr(clean: np.ndarray, noisy: np.ndarray, max_val: float = 255.0) -> float:
    """Computes Peak Signal-to-Noise Ratio (PSNR) between clean and noisy images.

    Args:
        clean (np.ndarray): Ground truth target image (uint8 or float).
        noisy (np.ndarray): Noisy input image (uint8 or float).
        max_val (float): Maximum dynamic range value (e.g., 255.0 for uint8, 1.0 for normalized floats).

    Returns:
        float: PSNR in decibels (dB). Returns inflection upper limit (100.0) if images are identical.
    """
    mse = calculate_mse(clean, noisy)
    if mse == 0:
        return 100.0 # Identical signals limit
    
    psnr = 10.0 * np.log10((max_val ** 2) / mse)
    return float(psnr)


def calculate_ssim(
    img1: np.ndarray,
    img2: np.ndarray,
    max_val: float = 255.0,
    k1: float = 0.01,
    k2: float = 0.03,
    win_size: int = 11,
) -> float:
    """Computes Structural Similarity Index (SSIM) between twin images.

    Implements the standard SSIM equation using a 2D Gaussian window to evaluate local
    structure, contrast, and brightness. This matches scikit-image's implementation
    while maintaining zero dependencies.

    Args:
        img1 (np.ndarray): Gray image 1 (uint8 or float).
        img2 (np.ndarray): Gray image 2 (uint8 or float).
        max_val (float): Range of pixel values (255.0 for uint8, 1.0 for float).
        k1 (float): SSIM hyperparameter for luminance stabilization.
        k2 (float): SSIM hyperparameter for contrast stabilization.
        win_size (int): Gaussian filter kernel width.

    Returns:
        float: SSIM index ranging [-1.0, 1.0] (higher is better).
    """
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2

    # Standardize image formats to float64
    x = img1.astype(np.float64)
    y = img2.astype(np.float64)

    # Re-evaluate window width parameter
    if x.shape[0] < win_size or x.shape[1] < win_size:
        win_size = min(x.shape[0], x.shape[1])
        if win_size % 2 == 0:
            win_size = max(1, win_size - 1)

    # Establish Gaussian filter kernel
    # sigma = 1.5 standard deviation is default for win_size=11
    sigma = 1.5
    gaussian_kernel = cv2.getGaussianKernel(win_size, sigma)
    window = np.outer(gaussian_kernel, gaussian_kernel)

    # Calculate local means
    mu_x = cv2.filter2D(x, -1, window)[win_size//2 : -win_size//2, win_size//2 : -win_size//2]
    mu_y = cv2.filter2D(y, -1, window)[win_size//2 : -win_size//2, win_size//2 : -win_size//2]

    # Squares of local means
    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    # Local variances and covariance
    sigma_x_sq = cv2.filter2D(x ** 2, -1, window)[win_size//2 : -win_size//2, win_size//2 : -win_size//2] - mu_x_sq
    sigma_y_sq = cv2.filter2D(y ** 2, -1, window)[win_size//2 : -win_size//2, win_size//2 : -win_size//2] - mu_y_sq
    sigma_xy = cv2.filter2D(x * y, -1, window)[win_size//2 : -win_size//2, win_size//2 : -win_size//2] - mu_xy

    # SSIM equation components
    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)

    ssim_map = numerator / denominator
    return float(np.mean(ssim_map))


def calculate_relative_snr(clean: np.ndarray, noisy: np.ndarray) -> float:
    """Computes reference-based Signal-to-Noise Ratio (SNR) in dB.

    Defined as 10 * log10(Power of Signal / Power of Noise).
    Where Signal power is computed from the clean ground truth and Noise power is the MSE.

    Args:
        clean (np.ndarray): Reference ground truth image values.
        noisy (np.ndarray): Noisy processed image values.

    Returns:
        float: SNR score in decibels (dB).
    """
    c_arr = clean.astype(np.float64)
    n_arr = noisy.astype(np.float64)
    
    # Noise residue estimation
    noise = c_arr - n_arr
    
    power_signal = np.mean(c_arr ** 2)
    power_noise = np.mean(noise ** 2)

    if power_noise == 0:
        return 100.0 # Upper limit
    if power_signal == 0:
        return -100.0 # Standard lower limit

    return float(10.0 * np.log10(power_signal / power_noise))


def calculate_image_snr(image: np.ndarray) -> float:
    """Computes self-contained Signal-to-Noise Ratio (SNR) for a single image.

    Using the local statistical quotient: Mean / Standard Deviation.
    Commonly used for SEM metrology verification in uniform regions.

    Args:
        image (np.ndarray): Gray target image.

    Returns:
        float: SNR value in dB.
    """
    img_arr = image.astype(np.float64)
    mean = np.mean(img_arr)
    std = np.std(img_arr)

    if std == 0:
        return 100.0 # Absolutely uniform signal
    
    snr = mean / std
    # Convert to decibel power ratio
    return float(20.0 * np.log10(snr)) if snr > 0 else -100.0
