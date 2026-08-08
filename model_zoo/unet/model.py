"""
U-Net model implementation for Semiconductor Image Restoration (Module 6).
Inherits from BaseModel and implements an encoder-decoder architecture with skip connections.
"""

import torch
import torch.nn as nn
from model_zoo.common.base_model import BaseModel


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class UNet(BaseModel):
    """Standard U-Net architecture adapted for single-channel semiconductor images."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, init_features: int = 32) -> None:
        super().__init__()
        self.inc = DoubleConv(in_channels, init_features)
        
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(init_features, init_features * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(init_features * 2, init_features * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(init_features * 4, init_features * 8))
        
        self.up1 = nn.ConvTranspose2d(init_features * 8, init_features * 4, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(init_features * 8, init_features * 4)
        
        self.up2 = nn.ConvTranspose2d(init_features * 4, init_features * 2, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(init_features * 4, init_features * 2)
        
        self.up3 = nn.ConvTranspose2d(init_features * 2, init_features, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(init_features * 2, init_features)
        
        self.outc = nn.Conv2d(init_features, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        u1 = self.up1(x4)
        # Skip connection concat
        u1_cat = torch.cat([u1, x3], dim=1)
        u1_conv = self.conv_up1(u1_cat)
        
        u2 = self.up2(u1_conv)
        u2_cat = torch.cat([u2, x2], dim=1)
        u2_conv = self.conv_up2(u2_cat)
        
        u3 = self.up3(u2_conv)
        u3_cat = torch.cat([u3, x1], dim=1)
        u3_conv = self.conv_up3(u3_cat)
        
        logits = self.outc(u3)
        return logits
