import torch
import torch.nn as nn
from model_zoo.common.base_model import BaseModel

class DnCNN(BaseModel):
    """DnCNN model for noise residual learning."""
    def __init__(self, in_channels: int = 1, out_channels: int = 1, num_layers: int = 15, num_features: int = 64) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        ]
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(num_features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual learning: predicts noise
        noise = self.dncnn(x)
        return x - noise
