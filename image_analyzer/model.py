"""
Physics-Guided Image Analyzer neural network architecture (Module 7).
Implements a Multi-Task Learning (MTL) architecture with a shared backbone and specialized heads.
"""

import math
from typing import Dict, Tuple, Any
import torch
import torch.nn as nn
import torchvision.models


class PhysicsImageAnalyzer(nn.Module):
    """Multi-Task degradation feature extractor and prediction network."""

    def __init__(self, backbone_name: str = "efficientnet_b0") -> None:
        """Initializes the multi-task network.

        Args:
            backbone_name (str): 'efficientnet_b0', 'mobilenet_v3', or 'convnext_tiny'.
        """
        super().__init__()
        self.backbone_name = backbone_name

        # Load shared backbone feature extractor
        if backbone_name == "efficientnet_b0":
            backbone = torchvision.models.efficientnet_b0(weights=None)
            self.features = backbone.features
            self.num_features = 1280
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif backbone_name == "mobilenet_v3":
            backbone = torchvision.models.mobilenet_v3_small(weights=None)
            self.features = backbone.features
            self.num_features = 576
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif backbone_name == "convnext_tiny":
            backbone = torchvision.models.convnext_tiny(weights=None)
            # ConvNeXt features ends with a block of layers, let's extract the feature representation
            self.features = backbone.features
            self.num_features = 768
            self.pool = nn.AdaptiveAvgPool2d(1)
        else:
            raise ValueError(f"Unknown backbone choice: {backbone_name}")

        # Shared feature projection head
        self.shared_fc = nn.Sequential(
            nn.Linear(self.num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Classification prediction heads
        self.noise_type_head = nn.Linear(256, 4)      # none, poisson_gaussian, sensor, e-beam_shot_noise
        self.blur_type_head = nn.Linear(256, 4)       # none, gaussian, defocus, motion
        self.severity_head = nn.Linear(256, 4)        # Easy, Medium, Hard, Extreme

        # Regression prediction head (outputs 11 continuous attributes)
        # 0. noise_level
        # 1. blur_strength
        # 2. resolution_loss
        # 3. compression_quality
        # 4. brightness
        # 5. contrast
        # 6. gamma
        # 7. edge_density
        # 8. texture_complexity
        # 9. entropy
        # 10. confidence
        self.regression_head = nn.Linear(256, 11)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Runs multi-task forward propagation.

        Args:
            x (torch.Tensor): Greyscale input image tensor, shape [B, 1, H, W]

        Returns:
            Dict[str, torch.Tensor]: Dictionary of output classification logits and continuous regressions.
        """
        # Convert single channel input [B, 1, H, W] -> [B, 3, H, W] to match pretrained RGB backbones
        x_rgb = torch.cat([x, x, x], dim=1)

        # Extract features
        feats = self.features(x_rgb)
        feats = self.pool(feats)
        feats = torch.flatten(feats, 1)

        # Shared project FC layer
        projected = self.shared_fc(feats)

        # Classifications logits
        noise_type_logits = self.noise_type_head(projected)
        blur_type_logits = self.blur_type_head(projected)
        severity_logits = self.severity_head(projected)

        # Regressions
        reg_raw = self.regression_head(projected)

        # Apply appropriate physical activation functions to keep predictions inside natural bounds
        # Bounded between [0, 1] via Sigmoid
        noise_level = torch.sigmoid(reg_raw[:, 0:1])
        resolution_loss = torch.sigmoid(reg_raw[:, 2:3])
        compression_quality = torch.sigmoid(reg_raw[:, 3:4])
        brightness = torch.sigmoid(reg_raw[:, 4:5])
        contrast = torch.sigmoid(reg_raw[:, 5:6])
        edge_density = torch.sigmoid(reg_raw[:, 7:8])
        texture_complexity = torch.sigmoid(reg_raw[:, 8:9])
        entropy = torch.sigmoid(reg_raw[:, 9:10])
        confidence = torch.sigmoid(reg_raw[:, 10:11])

        # Positive-only (ReLU) unbounded values
        blur_strength = torch.relu(reg_raw[:, 1:2])
        gamma = torch.relu(reg_raw[:, 6:7]) + 1e-4 # Avoid absolute zero

        return {
            "noise_type": noise_type_logits,
            "blur_type": blur_type_logits,
            "severity": severity_logits,
            "noise_level": noise_level.squeeze(-1),
            "blur_strength": blur_strength.squeeze(-1),
            "resolution_loss": resolution_loss.squeeze(-1),
            "compression_quality": compression_quality.squeeze(-1),
            "brightness": brightness.squeeze(-1),
            "contrast": contrast.squeeze(-1),
            "gamma": gamma.squeeze(-1),
            "edge_density": edge_density.squeeze(-1),
            "texture_complexity": texture_complexity.squeeze(-1),
            "entropy": entropy.squeeze(-1),
            "confidence": confidence.squeeze(-1)
        }

    def get_parameter_count(self) -> int:
        """Returns total number of trainable model parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_flops(self, input_size: Tuple[int, int, int, int] = (1, 1, 128, 128)) -> int:
        """Mathematical estimation of forward FLOPs for the backbone."""
        # Simple FLOP estimation proxy matching model zoo common architecture
        params = self.get_parameter_count()
        _, _, h, w = input_size
        # Proxy multiplier based on standard resolution scaling
        return int(params * (h * w / 16384) * 2)
