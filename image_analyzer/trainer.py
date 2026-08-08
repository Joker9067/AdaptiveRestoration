"""
Trainer module for the Physics-Guided Image Analyzer (Module 7).
Handles multi-task classification and regression losses, AMP, checkpointing, and early stopping.
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from image_analyzer.config import AnalyzerConfig

logger = logging.getLogger(__name__)


class ImageAnalyzerTrainer:
    """Trainer class for PhysicsImageAnalyzer with multi-task losses."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        config: AnalyzerConfig,
        device: torch.device,
        model_name: str
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.model_name = model_name

        # Loss weights and parameters
        self.hparams = config.hyperparameters
        self.loss_weights = self.hparams.loss_weights
        
        # Loss definitions
        self.ce_loss = nn.CrossEntropyLoss()
        self.smooth_l1_loss = nn.SmoothL1Loss()
        self.mse_loss = nn.MSELoss()

        # Path setups
        self.checkpoint_dir = Path(config.checkpoint_dir) / model_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.tensorboard_dir = Path(config.tensorboard_dir) / model_name
        self.writer = SummaryWriter(log_dir=str(self.tensorboard_dir))

        self.log_file = self.checkpoint_dir / "training_log.csv"
        self.history: list = []

        # AMP Configuration
        self.use_amp = self.hparams.amp and self.device.type != "cpu"
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)

    def _compute_loss(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Computes the combined weighted multi-task loss."""
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=self.device)

        # 1. Classification Heads
        for key in ["noise_type", "blur_type", "severity"]:
            w = self.loss_weights.get(key, 1.0)
            loss_val = self.ce_loss(outputs[key], targets[key].to(self.device))
            loss_dict[key] = loss_val.item()
            total_loss = total_loss + (w * loss_val)

        # 2. Regression Heads
        # Smooth L1 for noise_level, blur_strength, resolution_loss, and confidence
        # MSE for other continuous properties
        regression_tasks = {
            "noise_level": "smooth_l1",
            "blur_strength": "smooth_l1",
            "resolution_loss": "smooth_l1",
            "compression_quality": "mse",
            "brightness": "mse",
            "contrast": "mse",
            "gamma": "mse",
            "edge_density": "mse",
            "texture_complexity": "mse",
            "entropy": "mse",
            "confidence": "smooth_l1"
        }

        for key, loss_type in regression_tasks.items():
            w = self.loss_weights.get(key, 1.0)
            pred = outputs[key]
            tgt = targets[key].to(self.device)
            
            if loss_type == "smooth_l1":
                loss_val = self.smooth_l1_loss(pred, tgt)
            else:
                loss_val = self.mse_loss(pred, tgt)
                
            loss_dict[key] = loss_val.item()
            total_loss = total_loss + (w * loss_val)

        loss_dict["total"] = total_loss.item()
        return total_loss, loss_dict

    def _train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        epoch_losses: Dict[str, float] = {}

        for batch_idx, (inputs, _, targets) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            self.optimizer.zero_grad()

            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(inputs)
                loss, batch_losses = self._compute_loss(outputs, targets)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Accumulate losses
            for k, v in batch_losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v

        num_batches = len(self.train_loader)
        avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
        return avg_losses

    def _validate(self) -> Dict[str, float]:
        self.model.eval()
        val_losses: Dict[str, float] = {}

        with torch.no_grad():
            for inputs, _, targets in self.val_loader:
                inputs = inputs.to(self.device)
                
                with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    outputs = self.model(inputs)
                    _, batch_losses = self._compute_loss(outputs, targets)

                for k, v in batch_losses.items():
                    val_losses[k] = val_losses.get(k, 0.0) + v

        num_batches = len(self.val_loader)
        avg_val_losses = {k: v / num_batches for k, v in val_losses.items()}
        return avg_val_losses

    def train(self) -> Dict[str, Any]:
        """Runs the main training and validation loops."""
        logger.info(f"Starting training run for Image Analyzer model '{self.model_name}' on {self.device}...")
        
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.hparams.epochs):
            t0 = time.time()
            train_losses = self._train_one_epoch(epoch)
            val_losses = self._validate()
            epoch_time = time.time() - t0

            # Learning rate updates
            if self.scheduler:
                self.scheduler.step()

            # Logging to TensorBoard
            for k, v in train_losses.items():
                self.writer.add_scalar(f"Loss/Train_{k}", v, epoch)
            for k, v in val_losses.items():
                self.writer.add_scalar(f"Loss/Val_{k}", v, epoch)
            self.writer.add_scalar("Meta/EpochTime", epoch_time, epoch)

            # Logging progress
            logger.info(
                f"Epoch [{epoch + 1}/{self.hparams.epochs}] - "
                f"Train Loss: {train_losses['total']:.4f} | Val Loss: {val_losses['total']:.4f} | "
                f"Time: {epoch_time:.2f}s"
            )

            # Record history
            history_entry = {
                "epoch": epoch,
                "train_loss": train_losses["total"],
                "val_loss": val_losses["total"],
                "time_sec": epoch_time
            }
            self.history.append(history_entry)

            # Save check-points
            val_loss = val_losses["total"]
            state = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
                "config": self.config
            }
            torch.save(state, self.checkpoint_dir / "last.pt")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(state, self.checkpoint_dir / "best.pt")
                patience_counter = 0
                logger.info(f"--> Saved new best checkpoint at epoch {epoch + 1}")
            else:
                patience_counter += 1

            if patience_counter >= self.hparams.early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch + 1}.")
                break

        # Save history to CSV
        df = pd.DataFrame(self.history)
        df.to_csv(self.log_file, index=False)
        self.writer.close()

        return {
            "model_name": self.model_name,
            "best_val_loss": best_val_loss,
            "epochs_completed": len(self.history)
        }
