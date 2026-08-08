import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_zoo.common.base_model import BaseModel

class MDTA(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.qkv = nn.Conv2d(c, c * 3, 1)
        self.qkv_dw = nn.Conv2d(c * 3, c * 3, 3, padding=1, groups=c * 3)
        self.project = nn.Conv2d(c, c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_dw(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        
        b, c, h, w = q.shape
        q_flat = q.reshape(b, c, h * w)
        k_flat = k.reshape(b, c, h * w).transpose(-1, -2)
        v_flat = v.reshape(b, c, h * w)
        
        attn = (q_flat @ k_flat) * (1.0 / math.sqrt(c))
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v_flat).reshape(b, c, h, w)
        return x + self.project(out)

class GDFN(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c, c * 2, 1)
        self.dwconv = nn.Conv2d(c * 2, c * 2, 3, padding=1, groups=c * 2)
        self.project = nn.Conv2d(c, c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.dwconv(self.conv(x)).chunk(2, dim=1)
        gate = x1 * x2
        return x + self.project(gate)

class RestormerBlock(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.attn = MDTA(c)
        self.ffn = GDFN(c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(x)
        x = self.ffn(x)
        return x

class Restormer(BaseModel):
    """Restormer Transformer architecture."""
    def __init__(self, in_channels: int = 1, out_channels: int = 1, num_features: int = 32, num_blocks: int = 3) -> None:
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, num_features, 3, padding=1)
        self.blocks = nn.Sequential(*[RestormerBlock(num_features) for _ in range(num_blocks)])
        self.conv_out = nn.Conv2d(num_features, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_in(x)
        res = self.blocks(feat)
        out = self.conv_out(res + feat)
        return out
