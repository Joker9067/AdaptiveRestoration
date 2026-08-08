import torch
import torch.nn as nn
from model_zoo.common.base_model import BaseModel

class ChannelAttention(nn.Module):
    def __init__(self, num_features: int, reduction: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(num_features, num_features // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // reduction, num_features, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class EAMBlock(nn.Module):
    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.ca = ChannelAttention(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.relu1(self.conv1(x))
        res = self.conv2(res)
        res = self.ca(res)
        return x + res

class RIDNet(BaseModel):
    """Residual in Residual network with Channel Attention."""
    def __init__(self, in_channels: int = 1, out_channels: int = 1, num_features: int = 32, num_blocks: int = 3) -> None:
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, num_features, 3, padding=1)
        self.blocks = nn.ModuleList([EAMBlock(num_features) for _ in range(num_blocks)])
        self.conv_out = nn.Conv2d(num_features, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_in(x)
        res = feat
        for block in self.blocks:
            res = block(res)
        out = self.conv_out(res + feat)
        return out
