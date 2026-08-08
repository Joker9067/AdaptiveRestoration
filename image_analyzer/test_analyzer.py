"""
Unit and integration tests for the Physics-Guided Image Analyzer (Module 7).
Verifies:
1. Shape consistency and key attributes for all 3 backbones.
2. Target dictionary structure of ImageAnalyzerDataset.
3. Macro classification and regression metric calculators correctness.
4. Model training step backward path execution.
"""

import unittest
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from image_analyzer.model import PhysicsImageAnalyzer
from image_analyzer.trainer import ImageAnalyzerTrainer
from image_analyzer.config import AnalyzerConfig
from image_analyzer.run_benchmarks import calculate_classification_metrics, calculate_regression_metrics


class TestImageAnalyzerModel(unittest.TestCase):
    """Unit tests checking output keys, shapes, parameter counts, and FLOP estimations."""

    def setUp(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dummy_input = torch.zeros((2, 1, 128, 128), device=self.device)

    def test_shapes_efficientnet(self) -> None:
        model = PhysicsImageAnalyzer(backbone_name="efficientnet_b0").to(self.device)
        out = model(self.dummy_input)
        
        # Verify classification keys
        self.assertEqual(out["noise_type"].shape, (2, 4))
        self.assertEqual(out["blur_type"].shape, (2, 4))
        self.assertEqual(out["severity"].shape, (2, 4))
        
        # Verify regression keys
        reg_keys = [
            "noise_level", "blur_strength", "resolution_loss", "compression_quality",
            "brightness", "contrast", "gamma", "edge_density", "texture_complexity", "entropy", "confidence"
        ]
        for key in reg_keys:
            self.assertEqual(out[key].shape, (2,))

        self.assertGreater(model.get_parameter_count(), 0)
        self.assertGreater(model.get_flops(input_size=(1, 1, 128, 128)), 0)

    def test_shapes_mobilenet(self) -> None:
        model = PhysicsImageAnalyzer(backbone_name="mobilenet_v3").to(self.device)
        out = model(self.dummy_input)
        self.assertEqual(out["noise_type"].shape, (2, 4))
        self.assertEqual(out["noise_level"].shape, (2,))

    def test_shapes_convnext(self) -> None:
        model = PhysicsImageAnalyzer(backbone_name="convnext_tiny").to(self.device)
        out = model(self.dummy_input)
        self.assertEqual(out["noise_type"].shape, (2, 4))
        self.assertEqual(out["noise_level"].shape, (2,))


class TestMetricCalculators(unittest.TestCase):
    """Unit tests verifying manual metrics math computations."""

    def test_classification_metrics(self) -> None:
        preds = np.array([0, 1, 2, 0])
        targets = np.array([0, 1, 2, 1])
        acc, prec, rec, f1 = calculate_classification_metrics(preds, targets, num_classes=3)
        
        self.assertEqual(acc, 0.75)
        self.assertGreaterEqual(prec, 0.0)
        self.assertGreaterEqual(f1, 0.0)

    def test_regression_metrics(self) -> None:
        preds = np.array([0.5, 0.8, 1.2])
        targets = np.array([0.6, 0.7, 1.0])
        mae, rmse, r2 = calculate_regression_metrics(preds, targets)
        
        self.assertAlmostEqual(mae, 0.13333333, places=5)
        self.assertGreater(rmse, 0.0)
        self.assertLessEqual(r2, 1.0)


class TestTrainerExecution(unittest.TestCase):
    """Integration tests verifying multi-task loss backward passes during training steps."""

    def test_step_loss_backward(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PhysicsImageAnalyzer(backbone_name="mobilenet_v3").to(device)

        # Create mock target variables dictionary dataloader
        class DummyMTLDataset(torch.utils.data.Dataset):
            def __len__(self) -> int:
                return 4
            def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
                x = torch.randn(1, 64, 64)
                y = {
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
                return x, x, y

        dataset = DummyMTLDataset()
        loader = DataLoader(dataset, batch_size=2)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        config = AnalyzerConfig(
            checkpoint_dir="./checkpoints/test_analyzer_run",
            tensorboard_dir="./runs/test_analyzer_run"
        )
        config.hyperparameters.epochs = 1
        
        trainer = ImageAnalyzerTrainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            optimizer=optimizer,
            scheduler=None,
            config=config,
            device=device,
            model_name="test_mobilenet"
        )

        res = trainer.train()
        self.assertEqual(res["model_name"], "test_mobilenet")

        # Clean check-points files
        import shutil
        if Path("./checkpoints/test_analyzer_run").exists():
            shutil.rmtree("./checkpoints/test_analyzer_run")
        if Path("./runs/test_analyzer_run").exists():
            shutil.rmtree("./runs/test_analyzer_run")


if __name__ == "__main__":
    unittest.main()
