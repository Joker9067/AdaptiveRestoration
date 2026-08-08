"""
Testing suite for Model Zoo (Module 6).
Verifies:
1. Instantiation and output shape consistency for all 8 models.
2. Parameter and FLOP capture hooks.
3. Charbonnier, SSIM, and Combined weighted loss calculations.
4. ModelTrainer dry-run training loop iterations.
"""

import unittest
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Import models
from model_zoo.dncnn.model import DnCNN
from model_zoo.ridnet.model import RIDNet
from model_zoo.nafnet.model import NAFNet
from model_zoo.edsr.model import EDSR
from model_zoo.rcan.model import RCAN
from model_zoo.swinir.model import SwinIR
from model_zoo.restormer.model import Restormer
from model_zoo.unet.model import UNet

# Import losses and trainers
from model_zoo.common.losses import CharbonnierLoss, SSIMLoss, CombinedWeightedLoss
from model_zoo.common.trainer import ModelTrainer


class TestModelZooArchitectures(unittest.TestCase):
    """Unit tests verifying shape propagation, parameter sizes, and FLOP estimations."""

    def setUp(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dummy_input = torch.zeros((1, 1, 128, 128), device=self.device)

    def test_shapes_noise_models(self) -> None:
        """Noise models must preserve shape exactly."""
        models = [DnCNN(), RIDNet(), NAFNet()]
        for m in models:
            m = m.to(self.device)
            out = m(self.dummy_input)
            self.assertEqual(out.shape, self.dummy_input.shape)
            self.assertGreater(m.get_parameter_count(), 0)
            self.assertGreater(m.get_flops(input_size=(1, 1, 128, 128)), 0)

    def test_shapes_sr_models(self) -> None:
        """Super resolution models must scale input size internally and return matched shape output."""
        models = [EDSR(upscale_factor=2), RCAN(upscale_factor=2), SwinIR(upscale_factor=2)]
        for m in models:
            m = m.to(self.device)
            out = m(self.dummy_input)
            self.assertEqual(out.shape, self.dummy_input.shape)
            self.assertGreater(m.get_parameter_count(), 0)

    def test_shapes_structure_models(self) -> None:
        """Structure restoration models (Restormer, U-Net) shape matches."""
        models = [Restormer(), UNet()]
        for m in models:
            m = m.to(self.device)
            out = m(self.dummy_input)
            self.assertEqual(out.shape, self.dummy_input.shape)
            self.assertGreater(m.get_parameter_count(), 0)


class TestLosses(unittest.TestCase):
    """Unit tests for configurable losses."""

    def setUp(self) -> None:
        self.pred = torch.ones((1, 1, 64, 64)) * 0.8
        self.target = torch.ones((1, 1, 64, 64)) * 0.9

    def test_charbonnier(self) -> None:
        loss_fn = CharbonnierLoss(eps=1e-3)
        loss = loss_fn(self.pred, self.target)
        self.assertGreater(loss.item(), 0.0)

    def test_ssim_loss(self) -> None:
        loss_fn = SSIMLoss(window_size=5, sigma=1.0)
        loss = loss_fn(self.pred, self.target)
        self.assertGreater(loss.item(), 0.0)

    def test_combined_weighted_loss(self) -> None:
        loss_weights = {"l1": 1.0, "ssim": 0.5, "charbonnier": 0.2}
        loss_fn = CombinedWeightedLoss(loss_weights)
        loss = loss_fn(self.pred, self.target)
        self.assertGreater(loss.item(), 0.0)


class TestTrainerDryRun(unittest.TestCase):
    """Integration test checking training loop execution path."""

    def test_dry_run_epoch(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DnCNN(num_layers=4, num_features=8) # Small model for fast execution
        
        # Create dummy tensors dataset
        x = torch.randn(4, 1, 64, 64)
        y = torch.randn(4, 1, 64, 64)
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=2)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        loss_fn = CombinedWeightedLoss({"l1": 1.0})
        
        cfg = {
            "epochs": 1,
            "early_stopping_patience": 1,
            "amp": True,
            "checkpoint_dir": "./checkpoints/test_run",
            "tensorboard_dir": "./runs/test_run"
        }

        trainer = ModelTrainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            optimizer=optimizer,
            scheduler=None,
            loss_fn=loss_fn,
            config=cfg,
            device=device,
            model_name="test_dncnn"
        )
        
        meta = trainer.train()
        self.assertEqual(meta["model_name"], "test_dncnn")
        self.assertTrue(Path("./checkpoints/test_run/test_dncnn/best.pt").exists())

        # Cleanup test files
        import shutil
        if Path("./checkpoints/test_run").exists():
            shutil.rmtree("./checkpoints/test_run")
        if Path("./runs/test_run").exists():
            shutil.rmtree("./runs/test_run")


if __name__ == "__main__":
    unittest.main()
