"""
Trainer module for the Adaptive Decision Engine (Module 8).
Trains the gating network and fusion block end-to-end using optimal sequence search supervision
and differentiable weighted attention fusion losses.
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

from decision_engine.config import DecisionEngineConfig
from decision_engine.model import ORDER_PERMUTATIONS
from model_zoo.common.losses import SSIMLoss

logger = logging.getLogger(__name__)


class DecisionEngineTrainer:
    """Trainer class for optimizing AdaptiveDecisionNet and AttentionFusionBlock."""

    def __init__(
        self,
        pipeline: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        config: DecisionEngineConfig,
        device: torch.device,
        model_name: str = "adaptive_pipeline"
    ) -> None:
        self.pipeline = pipeline.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.model_name = model_name

        self.hparams = config.hyperparameters
        self.loss_weights = self.hparams.loss_weights

        # Loss formulations
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss(window_size=7, sigma=1.5).to(device)
        self.ce_loss = nn.CrossEntropyLoss()

        # Path setups
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.tensorboard_dir = Path(config.tensorboard_dir) / model_name
        self.writer = SummaryWriter(log_dir=str(self.tensorboard_dir))

        self.log_file = self.checkpoint_dir / "training_log.csv"
        self.history: list = []

        # AMP Configuration
        self.use_amp = self.hparams.amp and self.device.type != "cpu"
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)

    def _find_optimal_permutation(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Finds the optimal permutation index (0-5) that yields the lowest L1 reconstruction error.

        Runs with gradient calculation disabled for experts to ensure fast CPU indexing search.
        """
        b = inputs.size(0)
        best_losses = torch.ones(b, device=self.device) * float("inf")
        best_indices = torch.zeros(b, dtype=torch.long, device=self.device)

        with torch.no_grad():
            for idx in range(len(ORDER_PERMUTATIONS)):
                # Evaluate full pipeline with forced order permutation
                fused, _, _, _ = self.pipeline(inputs, forced_order_idx=idx)
                
                # Calculate sample-wise L1 loss
                sample_losses = torch.mean(torch.abs(fused - targets), dim=(1, 2, 3))
                
                # Check for improvement
                improved = sample_losses < best_losses
                best_losses = torch.where(improved, sample_losses, best_losses)
                best_indices = torch.where(improved, torch.tensor(idx, device=self.device), best_indices)

        return best_indices

    def _compute_losses(
        self,
        fused: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor,
        order_logits: torch.Tensor,
        optimal_order_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Computes reconstruction, SSIM, routing order classification, and entropy regularization losses."""
        # 1. Reconstruction Losses (L1 + SSIM)
        l1_val = self.l1_loss(fused, targets)
        ssim_val = self.ssim_loss(fused, targets)

        # 2. Sequence Ordering Classification
        order_cls_val = self.ce_loss(order_logits, optimal_order_idx)

        # 3. Entropy Regularization: encourages diversity in routing weights
        # Avoid zero elements with small offset
        entropy_val = -torch.mean(torch.sum(weights * torch.log(weights + 1e-8), dim=1))

        # Combined loss
        w_recon = self.loss_weights.get("reconstruction", 1.0)
        w_ssim = self.loss_weights.get("ssim", 0.5)
        w_order = self.loss_weights.get("order_cls", 1.0)
        w_entropy = self.loss_weights.get("entropy_reg", 0.1)

        total_loss = (
            w_recon * l1_val +
            w_ssim * ssim_val +
            w_order * order_cls_val +
            w_entropy * entropy_val
        )

        loss_dict = {
            "reconstruction_l1": l1_val.item(),
            "ssim": ssim_val.item(),
            "order_cls": order_cls_val.item(),
            "entropy_reg": entropy_val.item(),
            "total": total_loss.item()
        }

        return total_loss, loss_dict

    def _train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.pipeline.gating_net.train()
        self.pipeline.fusion_block.train()
        epoch_losses: Dict[str, float] = {}

        for batch_idx, (inputs, gts, targets) in enumerate(self.train_loader):
            inputs, gts = inputs.to(self.device), gts.to(self.device)
            self.optimizer.zero_grad()

            # Find optimal permutation target index
            opt_indices = self._find_optimal_permutation(inputs, gts)

            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                # Run pipeline in forward mode
                fused, weights, _, order_logits = self.pipeline(inputs)
                loss, batch_losses = self._compute_losses(fused, gts, weights, order_logits, opt_indices)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            for k, v in batch_losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v

        num_batches = len(self.train_loader)
        return {k: v / num_batches for k, v in epoch_losses.items()}

    def _validate(self) -> Dict[str, float]:
        self.pipeline.gating_net.eval()
        self.pipeline.fusion_block.eval()
        val_losses: Dict[str, float] = {}

        with torch.no_grad():
            for inputs, gts, targets in self.val_loader:
                inputs, gts = inputs.to(self.device), gts.to(self.device)
                
                # Find optimal permutation target index
                opt_indices = self._find_optimal_permutation(inputs, gts)

                with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    fused, weights, _, order_logits = self.pipeline(inputs)
                    _, batch_losses = self._compute_losses(fused, gts, weights, order_logits, opt_indices)

                for k, v in batch_losses.items():
                    val_losses[k] = val_losses.get(k, 0.0) + v

        num_batches = len(self.val_loader)
        return {k: v / num_batches for k, v in val_losses.items()}

    def train(self) -> Dict[str, Any]:
        """Runs the main training and validation loops."""
        logger.info(f"Starting training run for Adaptive Decision Pipeline...")
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.hparams.epochs):
            t0 = time.time()
            train_losses = self._train_one_epoch(epoch)
            val_losses = self._validate()
            epoch_time = time.time() - t0

            if self.scheduler:
                self.scheduler.step()

            # Logging to TensorBoard
            for k, v in train_losses.items():
                self.writer.add_scalar(f"Loss/Train_{k}", v, epoch)
            for k, v in val_losses.items():
                self.writer.add_scalar(f"Loss/Val_{k}", v, epoch)
            self.writer.add_scalar("Meta/EpochTime", epoch_time, epoch)

            logger.info(
                f"Epoch [{epoch + 1}/{self.hparams.epochs}] - "
                f"Train Loss: {train_losses['total']:.4f} | Val Loss: {val_losses['total']:.4f} | "
                f"Time: {epoch_time:.2f}s"
            )

            history_entry = {
                "epoch": epoch,
                "train_loss": train_losses["total"],
                "val_loss": val_losses["total"],
                "time_sec": epoch_time
            }
            self.history.append(history_entry)

            val_loss = val_losses["total"]
            state = {
                "epoch": epoch,
                "model_state_dict": self.pipeline.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
                "config": self.config
            }
            torch.save(state, self.checkpoint_dir / "last_pipeline.pt")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(state, self.checkpoint_dir / "best_pipeline.pt")
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
