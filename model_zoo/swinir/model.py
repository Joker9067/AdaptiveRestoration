import torch
import torch.nn as nn
import torch.nn.functional as F
from model_zoo.common.base_model import BaseModel

class SwinTransformerBlock(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)
        
        self.sa = nn.Sequential(
            nn.Conv2d(c, c // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c // 4, 1, 1),
            nn.Sigmoid()
        )
        self.mlp = nn.Sequential(
            nn.Conv2d(c, c * 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c * 2, c, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.conv2(F.relu(self.conv1(x)))
        att = self.sa(res)
        res = res * att
        x = x + res
        return x + self.mlp(x)

class SwinIR(BaseModel):
    """SwinIR super resolution block model."""
    def __init__(self, in_channels: int = 1, out_channels: int = 1, num_features: int = 32, num_blocks: int = 3, upscale_factor: int = 2) -> None:
        super().__init__()
        self.upscale_factor = upscale_factor
        self.conv_in = nn.Conv2d(in_channels, num_features, 3, padding=1)
        self.blocks = nn.Sequential(*[SwinTransformerBlock(num_features) for _ in range(num_blocks)])
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
        res = self.conv_mid(self.blocks(feat))
        res = res + feat
        up = self.upsampler(res)
        out = self.conv_out(up)
        return out
