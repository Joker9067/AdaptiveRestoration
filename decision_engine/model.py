"""
Neural network architectures for the Adaptive Decision Engine (Module 8).
Implements the gating network, learnable attention fusion, and sequential pipeline.
"""

from typing import Dict, Tuple, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F

# Permutation mapping of execution orders
ORDER_PERMUTATIONS = [
    ("noise", "struct", "sr"),
    ("noise", "sr", "struct"),
    ("struct", "noise", "sr"),
    ("struct", "sr", "noise"),
    ("sr", "noise", "struct"),
    ("sr", "struct", "noise")
]


class AdaptiveDecisionNet(nn.Module):
    """Gating network predicting expert weights and processing orders from degradation properties."""

    def __init__(self, input_dim: int = 139) -> None:
        """Initializes the gating network.

        Args:
            input_dim (int): 11 (continuous parameters) + 128 (degradation embedding) = 139.
        """
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Output 1: normalized weights for 3 experts (Noise, SR, Structure)
        self.weights_head = nn.Sequential(
            nn.Linear(64, 3),
            nn.Softmax(dim=1)
        )
        
        # Output 2: probability distribution over 6 ordering permutations
        self.order_head = nn.Linear(64, 6)

    def forward(self, degradation_vector: torch.Tensor, embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs the forward gating path.

        Args:
            degradation_vector (torch.Tensor): Continuous targets [B, 11]
            embedding (torch.Tensor): 128-dimensional image embedding [B, 128]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (weights [B, 3], order_logits [B, 6])
        """
        x = torch.cat([degradation_vector, embedding], dim=1)
        feats = self.fc(x)
        weights = self.weights_head(feats)
        order_logits = self.order_head(feats)
        return weights, order_logits


class AttentionFusionBlock(nn.Module):
    """Learnable feature-weighted channel attention and residual convolutional reconstruction fusion block."""

    def __init__(self) -> None:
        super().__init__()
        # Attention projection from 3 weights -> 3 scaling factors
        self.attn_proj = nn.Sequential(
            nn.Linear(3, 3),
            nn.Sigmoid()
        )
        
        # Conv reconstruction head
        self.reconstruct = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1)
        )

    def forward(self, x_stacked: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Fuses expert outputs using learnable channel attention.

        Args:
            x_stacked (torch.Tensor): Stacking of 3 expert restorations [B, 3, H, W]
            weights (torch.Tensor): Gating weights [B, 3]

        Returns:
            torch.Tensor: Differentiably fused grayscale image [B, 1, H, W]
        """
        b, c, h, w = x_stacked.shape
        
        # Get channel-wise attention scaling weights
        attn_scales = self.attn_proj(weights).view(b, c, 1, 1)
        x_scaled = x_stacked * attn_scales
        
        # Conv reconstruction path
        reconstructed = self.reconstruct(x_scaled)
        
        # Residual connection to the baseline weighted average to ensure training stability
        baseline_weighted = (
            weights[:, 0].view(b, 1, 1, 1) * x_stacked[:, 0:1] +
            weights[:, 1].view(b, 1, 1, 1) * x_stacked[:, 1:2] +
            weights[:, 2].view(b, 1, 1, 1) * x_stacked[:, 2:3]
        )
        
        return reconstructed + baseline_weighted


class RestorationPipeline(nn.Module):
    """End-to-End Restoration Pipeline incorporating sequential routing and attention fusion."""

    def __init__(
        self,
        expert_noise: nn.Module,
        expert_sr: nn.Module,
        expert_struct: nn.Module,
        analyzer: nn.Module
    ) -> None:
        """Initializes the pipeline. Gating and fusion are active; experts and analyzer are frozen.

        Args:
            expert_noise (nn.Module): Pre-trained Noise Expert.
            expert_sr (nn.Module): Pre-trained SR Expert.
            expert_struct (nn.Module): Pre-trained Structure Expert.
            analyzer (nn.Module): Pre-trained Image Analyzer.
        """
        super().__init__()
        # Freeze non-trainable components
        self.expert_noise = expert_noise.eval()
        self.expert_sr = expert_sr.eval()
        self.expert_struct = expert_struct.eval()
        self.analyzer = analyzer.eval()
        for p in self.expert_noise.parameters():
            p.requires_grad = False
        for p in self.expert_sr.parameters():
            p.requires_grad = False
        for p in self.expert_struct.parameters():
            p.requires_grad = False
        for p in self.analyzer.parameters():
            p.requires_grad = False

        # Learnable gating and fusion blocks
        self.gating_net = AdaptiveDecisionNet(input_dim=139)
        self.fusion_block = AttentionFusionBlock()

        # Helper projection for the 128-dimensional embedding from analyzer features
        self.embedding_proj = nn.Linear(analyzer.num_features, 128)

    def _get_degradation_inputs(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inferences the analyzer to retrieve predicted vectors and embedding."""
        with torch.no_grad():
            x_rgb = torch.cat([x, x, x], dim=1)
            feats = self.analyzer.features(x_rgb)
            feats = self.analyzer.pool(feats)
            feats = torch.flatten(feats, 1)
            
            # Predict degradation attributes using analyzer model heads
            outputs = self.analyzer(x)
            
        # Re-pack continuous regression attributes to tensor [B, 11]
        keys = [
            "noise_level", "blur_strength", "resolution_loss", "compression_quality",
            "brightness", "contrast", "gamma", "edge_density", "texture_complexity", "entropy", "confidence"
        ]
        deg_vector = torch.stack([outputs[k] for k in keys], dim=1)
        
        # Project analyzer features to 128-dimensional embedding
        embedding = self.embedding_proj(feats)
        return deg_vector, embedding

    def _execute_expert_step(self, expert_name: str, x: torch.Tensor) -> torch.Tensor:
        """Forwards input tensor through the named expert."""
        if expert_name == "noise":
            return self.expert_noise(x)
        elif expert_name == "sr":
            return self.expert_sr(x)
        elif expert_name == "struct":
            return self.expert_struct(x)
        else:
            raise ValueError(f"Unknown expert: {expert_name}")

    def forward(self, x: torch.Tensor, forced_order_idx: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        """Runs the pipeline.

        Args:
            x (torch.Tensor): Degraded input image tensor [B, 1, H, W]
            forced_order_idx (int, optional): Forces a specific permutation during optimal-search steps.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
                - Fused restored image [B, 1, H, W]
                - Expert weights [B, 3]
                - Selected permutation index [0-5]
                - Order logits [B, 6]
        """
        # 1. Analyze degradation state
        deg_vector, embedding = self._get_degradation_inputs(x)

        # 2. Predict routing weights and ordering logits
        weights, order_logits = self.gating_net(deg_vector, embedding)

        # 3. Resolve execution order
        if forced_order_idx is not None:
            order_idx = forced_order_idx
        else:
            order_idx = int(torch.argmax(order_logits, dim=1)[0].item())
            
        permutation = ORDER_PERMUTATIONS[order_idx]

        # 4. Sequential execution of the selected permutation
        out1 = self._execute_expert_step(permutation[0], x)
        out2 = self._execute_expert_step(permutation[1], out1)
        out3 = self._execute_expert_step(permutation[2], out2)

        # Sort the intermediate outputs back to the canonical experts map channel order:
        # [Noise_output, SR_output, Structure_output] for standard input to the fusion block.
        channel_map = {permutation[i]: i for i in range(3)}
        
        noise_out = out1 if channel_map["noise"] == 0 else (out2 if channel_map["noise"] == 1 else out3)
        sr_out = out1 if channel_map["sr"] == 0 else (out2 if channel_map["sr"] == 1 else out3)
        struct_out = out1 if channel_map["struct"] == 0 else (out2 if channel_map["struct"] == 1 else out3)

        # 5. Differentiable attention fusion
        x_stacked = torch.cat([noise_out, sr_out, struct_out], dim=1)
        fused = self.fusion_block(x_stacked, weights)

        return fused, weights, order_idx, order_logits
