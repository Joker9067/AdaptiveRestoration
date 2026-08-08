import torch
import torch.nn as nn
from model_zoo.common.base_model import BaseModel

class NAFBlock(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(c)
        self.conv1 = nn.Conv2d(c, c * 2, 1)
        self.conv2 = nn.Conv2d(c * 2, c * 2, 3, padding=1, groups=c * 2)
        # Multiplication gate (Activation Free)
        self.conv3 = nn.Conv2d(c, c, 1)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Permute for LayerNorm
        b, c, h, w = x.shape
        res = x
        x_norm = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_conv = self.conv2(self.conv1(x_norm))
        # Split channels for multiplication gate
        x1, x2 = x_conv.chunk(2, dim=1)
        x_gate = x1 * x2
        x_att = self.conv3(x_gate)
        x_att = x_att * self.ca(x_att)
        return res + x_att

class NAFNet(BaseModel):
    """Nonlinear Activation Free Network."""
    def __init__(self, in_channels: int = 1, out_channels: int = 1, num_features: int = 32, num_blocks: int = 4) -> None:
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, num_features, 3, padding=1)
        self.blocks = nn.Sequential(*[NAFBlock(num_features) for _ in range(num_blocks)])
        self.conv_out = nn.Conv2d(num_features, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_in(x)
        res = self.blocks(feat)
        out = self.conv_out(res + feat)
        return out
