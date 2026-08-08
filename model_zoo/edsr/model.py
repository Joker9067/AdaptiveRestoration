import torch
import torch.nn as nn
import torch.nn.functional as F
from model_zoo.common.base_model import BaseModel

class ResBlock(nn.Module):
    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, 3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x)

class EDSR(BaseModel):
    """Enhanced Deep Super-Resolution Network without BN."""
    def __init__(self, in_channels: int = 1, out_channels: int = 1, num_features: int = 32, num_blocks: int = 8, upscale_factor: int = 2) -> None:
        super().__init__()
        self.upscale_factor = upscale_factor
        self.conv_in = nn.Conv2d(in_channels, num_features, 3, padding=1)
        self.res_blocks = nn.Sequential(*[ResBlock(num_features) for _ in range(num_blocks)])
        
        self.upsampler = nn.Sequential(
            nn.Conv2d(num_features, num_features * (upscale_factor ** 2), 3, padding=1),
            nn.PixelShuffle(upscale_factor)
        )
        self.conv_out = nn.Conv2d(num_features, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_sub = F.interpolate(x, size=(h // self.upscale_factor, w // self.upscale_factor), mode="bicubic", align_corners=False)
        
        feat = self.conv_in(x_sub)
        res = self.res_blocks(feat)
        res = res + feat
        up = self.upsampler(res)
        out = self.conv_out(up)
        return out
