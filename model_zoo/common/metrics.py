"""
Metrics calculation module for the Semiconductor Image Restoration System (Module 6).
Implements standard PSNR, SSIM, and LPIPS metrics using PyTorch and skimage.
"""

import logging
import math
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as skimage_ssim

logger = logging.getLogger(__name__)

# Global placeholder for LPIPS evaluator to avoid re-instantiation
_lpips_evaluator = None
_lpips_failed = False

def calculate_psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Calculates Peak Signal-to-Noise Ratio (PSNR) for tensors in [0, 1] range.

    Args:
        pred (torch.Tensor): Restored image tensor.
        gt (torch.Tensor): Ground truth clean image.

    Returns:
        float: Average PSNR in dB.
    """
    mse = F.mse_loss(pred, gt).item()
    if mse < 1e-10:
        return 100.0
    return float(20 * math.log10(1.0 / math.sqrt(mse)))


def calculate_ssim(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Calculates Structural Similarity Index Measure (SSIM) using skimage.

    Args:
        pred (torch.Tensor): Restored image tensor [B, C, H, W] in [0, 1].
        gt (torch.Tensor): Ground truth image tensor.

    Returns:
        float: Average SSIM.
    """
    p_np = pred.detach().cpu().numpy()
    g_np = gt.detach().cpu().numpy()

    batch_size = p_np.shape[0]
    ssim_sum = 0.0

    for i in range(batch_size):
        p_img = p_np[i].squeeze()
        g_img = g_np[i].squeeze()
        
        # Determine appropriate win_size
        min_dim = min(p_img.shape)
        win_size = min(7, min_dim) if min_dim < 7 else 7
        if win_size % 2 == 0:
            win_size -= 1

        score = skimage_ssim(
            g_img, p_img,
            data_range=1.0,
            win_size=win_size
        )
        ssim_sum += score

    return float(ssim_sum / batch_size)


def calculate_lpips(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Calculates Learned Perceptual Image Patch Similarity (LPIPS).

    Falls back gracefully to structural MSE-based error metric if offline
    or lpips is not installed.

    Args:
        pred (torch.Tensor): Restored image tensor [B, C, H, W] in [0, 1].
        gt (torch.Tensor): Ground truth image tensor.

    Returns:
        float: Average LPIPS score (lower is more perceptually similar).
    """
    global _lpips_evaluator, _lpips_failed
    
    # Grayscale to RGB duplication for LPIPS alexnet features
    if pred.size(1) == 1:
        pred = pred.expand(-1, 3, -1, -1)
        gt = gt.expand(-1, 3, -1, -1)

    # Normalize from [0, 1] to LPIPS expected range [-1, 1]
    pred_norm = 2.0 * pred - 1.0
    gt_norm = 2.0 * gt - 1.0

    if not _lpips_failed:
        try:
            if _lpips_evaluator is None:
                import lpips
                # Load LPIPS net='alex' in eval mode
                _lpips_evaluator = lpips.LPIPS(net="alex", verbose=False).to(pred.device).eval()
            
            with torch.no_grad():
                score = _lpips_evaluator(pred_norm, gt_norm)
                return float(torch.mean(score).item())
        except Exception as e:
            logger.warning(f"LPIPS calculation unavailable (offline weight download or module missing): {e}. "
                           "Falling back to visual MSE-based perceptual loss proxy.")
            _lpips_failed = True

    # Fallback perceptual score estimation based on (1 - SSIM) and L1
    l1 = F.l1_loss(pred, gt).item()
    ssim_val = calculate_ssim(pred, gt)
    fallback_score = 0.5 * (1.0 - ssim_val) + 0.5 * l1
    return float(fallback_score)
