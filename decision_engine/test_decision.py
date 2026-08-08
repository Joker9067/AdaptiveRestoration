"""
Unit and integration tests for the Adaptive Decision Engine (Module 8).
Verifies:
1. Gating weights normalization (summing to 1.0).
2. Differentiable attention fusion block.
3. Restoration sequential permutation execution paths.
4. Gating network and trainer optimization iterations.
"""

import unittest
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from decision_engine.model import AdaptiveDecisionNet, AttentionFusionBlock, RestorationPipeline, ORDER_PERMUTATIONS
from decision_engine.config import DecisionEngineConfig
from decision_engine.trainer import DecisionEngineTrainer


class TestDecisionEngineComponents(unittest.TestCase):
    """Unit tests checking model outputs, weight constraints, and shape properties."""

    def test_gating_weights(self) -> None:
        B = 4
        gating = AdaptiveDecisionNet(input_dim=139)
        deg_vector = torch.randn(B, 11)
        embedding = torch.randn(B, 128)
        
        weights, order_logits = gating(deg_vector, embedding)
        
        # Verify shape
        self.assertEqual(weights.shape, (B, 3))
        self.assertEqual(order_logits.shape, (B, 6))
        
        # Verify normalized weights sum to exactly 1.0
        sums = torch.sum(weights, dim=1)
        for val in sums:
            self.assertAlmostEqual(val.item(), 1.0, places=5)

    def test_attention_fusion(self) -> None:
        B, H, W = 2, 64, 64
        fusion = AttentionFusionBlock()
        x_stacked = torch.randn(B, 3, H, W)
        weights = torch.tensor([[0.5, 0.3, 0.2], [0.1, 0.8, 0.1]], dtype=torch.float32)
        
        out = fusion(x_stacked, weights)
        self.assertEqual(out.shape, (B, 1, H, W))


class TestRestorationPipelineExecution(unittest.TestCase):
    """Integration tests checking sequential expert routing and backpropagation path."""

    def test_pipeline_backward(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Define mock experts that preserve shape
        class MockExpert(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(1, 1, 3, padding=1)
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + self.conv(x)

        class MockAnalyzer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.num_features = 32
                self.conv = nn.Conv2d(3, 32, 3, padding=1)
                self.pool_layer = nn.AdaptiveAvgPool2d(1)
            def features(self, x: torch.Tensor) -> torch.Tensor:
                return self.conv(x)
            def pool(self, x: torch.Tensor) -> torch.Tensor:
                return self.pool_layer(x)
            def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
                b = x.shape[0]
                return {
                    "noise_level": torch.full((b,), 0.2, device=x.device),
                    "blur_strength": torch.full((b,), 0.5, device=x.device),
                    "resolution_loss": torch.full((b,), 0.1, device=x.device),
                    "compression_quality": torch.full((b,), 1.0, device=x.device),
                    "brightness": torch.full((b,), 0.5, device=x.device),
                    "contrast": torch.full((b,), 0.2, device=x.device),
                    "gamma": torch.full((b,), 1.0, device=x.device),
                    "edge_density": torch.full((b,), 0.3, device=x.device),
                    "texture_complexity": torch.full((b,), 0.4, device=x.device),
                    "entropy": torch.full((b,), 0.6, device=x.device),
                    "confidence": torch.full((b,), 1.0, device=x.device)
                }

        # Instantiate mock models
        exp_n = MockExpert().to(device)
        exp_sr = MockExpert().to(device)
        exp_s = MockExpert().to(device)
        analyzer = MockAnalyzer().to(device)

        pipeline = RestorationPipeline(
            expert_noise=exp_n,
            expert_sr=exp_sr,
            expert_struct=exp_s,
            analyzer=analyzer
        ).to(device)

        # Gating network parameter projections override
        pipeline.gating_net = AdaptiveDecisionNet(input_dim=11 + 128).to(device)
        pipeline.embedding_proj = nn.Linear(32, 128).to(device)

        # Mock MTL dataset yielding (inputs, targets)
        class DummyDataset(torch.utils.data.Dataset):
            def __len__(self) -> int:
                return 4
            def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
                x = torch.randn(1, 32, 32)
                y = torch.randn(1, 32, 32)
                targets = {
                    "noise_type": torch.tensor(0, dtype=torch.long),
                    "blur_type": torch.tensor(0, dtype=torch.long),
                    "severity": torch.tensor(0, dtype=torch.long),
                    "noise_level": torch.tensor(0.1, dtype=torch.float32),
                    "blur_strength": torch.tensor(0.5, dtype=torch.float32),
                    "resolution_loss": torch.tensor(0.2, dtype=torch.float32),
                    "compression_quality": torch.tensor(1.0, dtype=torch.float32),
                    "brightness": torch.tensor(0.5, dtype=torch.float32),
                    "contrast": torch.tensor(0.1, dtype=torch.float32),
                    "gamma": torch.tensor(1.0, dtype=torch.float32),
                    "edge_density": torch.tensor(0.3, dtype=torch.float32),
                    "texture_complexity": torch.tensor(0.4, dtype=torch.float32),
                    "entropy": torch.tensor(0.6, dtype=torch.float32),
                    "confidence": torch.tensor(1.0, dtype=torch.float32)
                }
                return x, y, targets

        dataset = DummyDataset()
        loader = DataLoader(dataset, batch_size=2)

        optimizer = torch.optim.Adam(
            list(pipeline.gating_net.parameters()) + list(pipeline.fusion_block.parameters()),
            lr=1e-4
        )
        config = DecisionEngineConfig(
            checkpoint_dir="./checkpoints/test_decision_run",
            tensorboard_dir="./runs/test_decision_run"
        )
        config.hyperparameters.epochs = 1

        trainer = DecisionEngineTrainer(
            pipeline=pipeline,
            train_loader=loader,
            val_loader=loader,
            optimizer=optimizer,
            scheduler=None,
            config=config,
            device=device,
            model_name="test_pipeline"
        )

        res = trainer.train()
        self.assertEqual(res["model_name"], "test_pipeline")

        # Cleanup checkpoints files
        import shutil
        if Path("./checkpoints/test_decision_run").exists():
            shutil.rmtree("./checkpoints/test_decision_run")
        if Path("./runs/test_decision_run").exists():
            shutil.rmtree("./runs/test_decision_run")


if __name__ == "__main__":
    unittest.main()
