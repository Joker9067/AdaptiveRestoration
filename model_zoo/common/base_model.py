"""
BaseModel abstract class for the Semiconductor Image Restoration System (Module 6).
All models in the Model Zoo inherit from this class to guarantee a common interface.
"""

from abc import ABC, abstractmethod
from typing import Tuple
import torch
import torch.nn as nn

class BaseModel(nn.Module, ABC):
    """Abstract Base Model class guaranteeing parameter and FLOPs measurement hooks."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass on the input tensor.

        Args:
            x (torch.Tensor): Input image tensor, shape [B, C, H, W] (float32).

        Returns:
            torch.Tensor: Output restored image tensor, shape [B, C, H, W] (float32).
        """
        pass

    def get_parameter_count(self) -> int:
        """Computes the total number of trainable parameters in the model.

        Returns:
            int: Trainable parameter count.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_flops(self, input_size: Tuple[int, int, int, int] = (1, 1, 512, 512)) -> int:
        """Estimates FLOPs (Floating Point Operations) of the model for a given input size.
        Calculates FLOPs for Conv2d and Linear layers mathematically based on shape propagation.

        Args:
            input_size (Tuple[int, int, int, int]): Shape of input tensor (B, C, H, W).

        Returns:
            int: Estimated FLOP count.
        """
        device = next(self.parameters()).device if list(self.parameters()) else torch.device("cpu")
        dummy_input = torch.zeros(input_size, device=device)
        flops = 0
        hooks = []

        def conv_hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            nonlocal flops
            x_in = inputs[0]
            batch_size = x_in.size(0)
            out_h, out_w = output.size(2), output.size(3)
            in_c = module.in_channels
            out_c = module.out_channels
            kh, kw = module.kernel_size
            
            # Multiply-Accumulate operations: 2 * in_c * out_c * kh * kw * out_h * out_w * batch_size
            layer_flops = 2 * batch_size * in_c * out_c * kh * kw * out_h * out_w
            # Include bias flops
            if module.bias is not None:
                layer_flops += batch_size * out_c * out_h * out_w
            flops += layer_flops

        def linear_hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            nonlocal flops
            x_in = inputs[0]
            batch_size = x_in.size(0)
            in_features = module.in_features
            out_features = module.out_features
            
            # Multiply-Accumulate operations: 2 * in_features * out_features * batch_size
            layer_flops = 2 * batch_size * in_features * out_features
            if module.bias is not None:
                layer_flops += batch_size * out_features
            flops += layer_flops

        # Register forward hooks
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                hooks.append(m.register_forward_hook(conv_hook))
            elif isinstance(m, nn.Linear):
                hooks.append(m.register_forward_hook(linear_hook))

        # Perform dry run forward pass to trigger hooks
        self.eval()
        with torch.no_grad():
            try:
                self.forward(dummy_input)
            except Exception as e:
                # Fallback to parameter count estimation if forward pass fails on dummy shape
                # standard estimation: parameter count * spatial size
                flops = self.get_parameter_count() * input_size[2] * input_size[3] // 10

        # Remove hooks
        for h in hooks:
            h.remove()

        return flops
