"""
Loss functions module for the Semiconductor Image Restoration System (Module 6).
Implements configurable L1, L2, Charbonnier, SSIM, VGG perceptual, and Combined weighted losses.
"""

import logging
from typing import Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class CharbonnierLoss(nn.Module):
    """Charbonnier loss function (robust L1 variant)."""

    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps2 = eps * eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.mean(torch.sqrt(diff * diff + self.eps2))
        return loss


class SSIMLoss(nn.Module):
    """Simplified structural similarity (SSIM) loss."""

    def __init__(self, window_size: int = 11, sigma: float = 1.5) -> None:
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.register_buffer("window", self._create_window(window_size, sigma))

    def _create_window(self, size: int, sigma: float) -> torch.Tensor:
        """Generates 1D Gaussian kernel and takes outer product to make 2D."""
        coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
        g = torch.exp(-coords**2 / (2.0 * sigma**2))
        g = g / g.sum()
        # Outer product to form 2D Gaussian kernel
        g2d = g.unsqueeze(1) @ g.unsqueeze(0)
        return g2d.unsqueeze(0).unsqueeze(0) # [1, 1, H, W]

    def _ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        channel = img1.size(1)
        window = self.window.expand(channel, 1, -1, -1).to(img1.device)
        
        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2
        
        c1 = 0.01**2
        c2 = 0.03**2
        
        ssim_num = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
        ssim_den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        
        ssim_val = ssim_num / ssim_den
        return torch.mean(ssim_val)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1.0 - self._ssim(pred, target)


class PerceptualLoss(nn.Module):
    """VGG16 feature map distance perceptual loss with mathematical fallback if offline."""

    def __init__(self) -> None:
        super().__init__()
        self.vgg = None
        self.fallback = False
        try:
            from torchvision.models import vgg16, VGG16_Weights
            # Load feature layers only
            vgg_model = vgg16(weights=VGG16_Weights.DEFAULT)
            self.vgg = nn.Sequential(*list(vgg_model.features)[:16]).eval()
            for p in self.vgg.parameters():
                p.requires_grad = False
            logger.info("Loaded VGG16 weights for Perceptual Loss successfully.")
        except Exception as e:
            logger.warning(f"Could not load VGG16 for perceptual loss (network offline or package missing): {e}. "
                           "Falling back to MSE loss calculation for this term.")
            self.fallback = True

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Grayscale to RGB duplication
        if pred.size(1) == 1:
            pred = pred.expand(-1, 3, -1, -1)
            target = target.expand(-1, 3, -1, -1)

        if self.fallback or self.vgg is None:
            return F.mse_loss(pred, target)
        
        pred_feat = self.vgg(pred)
        target_feat = self.vgg(target)
        return F.l1_loss(pred_feat, target_feat)


class CombinedWeightedLoss(nn.Module):
    """Weighted composition of multiple loss terms."""

    def __init__(self, loss_configs: Dict[str, float]) -> None:
        """Initializes with config mappings.

        Args:
            loss_configs (Dict[str, float]): Dict mapping loss names to weights.
                e.g. {"l1": 1.0, "ssim": 0.2, "perceptual": 0.05}
        """
        super().__init__()
        self.weights = loss_configs
        self.losses = nn.ModuleDict()

        for name in loss_configs.keys():
            key = name.lower()
            if key == "l1":
                self.losses[name] = nn.L1Loss()
            elif key in ("l2", "mse"):
                self.losses[name] = nn.MSELoss()
            elif key == "charbonnier":
                self.losses[name] = CharbonnierLoss()
            elif key == "ssim":
                self.losses[name] = SSIMLoss()
            elif key == "perceptual":
                self.losses[name] = PerceptualLoss()
            else:
                raise ValueError(f"Unknown loss term requested: '{name}'")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0
        for name, loss_fn in self.losses.items():
            weight = self.weights[name]
            if weight > 0:
                total_loss += weight * loss_fn(pred, target)
        return total_loss
