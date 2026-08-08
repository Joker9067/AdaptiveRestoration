import torch
import torch.nn as nn
import torch.nn.functional as F
from model_zoo.common.base_model import BaseModel

class RCAB(nn.Module):
    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_features, num_features, 3, padding=1)
        
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_features, num_features // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // 4, num_features, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.conv2(self.relu(self.conv1(x)))
        att = self.ca(res)
        return x + res * att

class ResidualGroup(nn.Module):
    def __init__(self, num_features: int, num_blocks: int) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*[RCAB(num_features) for _ in range(num_blocks)])
        self.conv = nn.Conv2d(num_features, num_features, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.conv(self.blocks(x))
        return x + res

class RCAN(BaseModel):
    """Residual Channel Attention Network."""
    def __init__(self, in_channels: int = 1, out_channels: int = 1, num_features: int = 32, num_groups: int = 2, num_blocks: int = 4, upscale_factor: int = 2) -> None:
        super().__init__()
        self.upscale_factor = upscale_factor
        self.conv_in = nn.Conv2d(in_channels, num_features, 3, padding=1)
        self.groups = nn.Sequential(*[ResidualGroup(num_features, num_blocks) for _ in range(num_groups)])
        self.conv_mid = nn.Conv2d(num_features, num_features, 3, padding=1)
        
        self.upsampler = nn.Sequential(
            nn.Conv2d(num_features, num_features * (upscale_factor ** 2), 3, padding=1),
            nn.PixelShuffle(upscale_factor)
        )
        self.conv_out = nn.Conv2d(num_features, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_sub = F.interpolate(x, size=(h // self.upscale_factor, w // self.upscale_factor), mode="bicubic", align_corners=False)
        
        feat = self.conv_in(x_sub)
        res = self.conv_mid(self.groups(feat))
        res = res + feat
        up = self.upsampler(res)
        out = self.conv_out(up)
        return out
