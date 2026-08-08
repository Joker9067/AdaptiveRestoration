"""
Trainer module for the Semiconductor Image Restoration System (Module 6).
Implements ModelTrainer supporting AMP, early stopping, gradient clipping, schedulers,
TensorBoard/CSV logging, and checkpoint saving/resuming with metadata.
"""

import os
import time
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model_zoo.common.metrics import calculate_psnr, calculate_ssim

logger = logging.getLogger(__name__)


def get_git_commit() -> str:
    """Attempts to retrieve the current git commit hash.

    Returns:
        str: Commit hash or 'unknown'.
    """
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return commit.decode("utf-8").strip()
    except Exception:
        return "unknown"


class ModelTrainer:
    """Unified Trainer for all restoration models in the Model Zoo."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        loss_fn: nn.Module,
        config: Dict[str, Any],
        device: torch.device,
        model_name: str,
        dataset_version: str = "1.0",
        seed: int = 42
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.config = config
        self.device = device
        self.model_name = model_name
        self.dataset_version = dataset_version
        self.seed = seed

        # Configuration variables
        self.epochs = int(config.get("epochs", 10))
        self.grad_clip = float(config.get("grad_clip", 5.0))
        self.use_amp = bool(config.get("amp", True))
        self.patience = int(config.get("early_stopping_patience", 5))

        # Checkpoints outputs paths
        self.checkpoint_dir = Path(config.get("checkpoint_dir", "./checkpoints")) / model_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_pth = self.checkpoint_dir / "best.pt"
        self.last_pth = self.checkpoint_dir / "last.pt"
        self.csv_log_pth = self.checkpoint_dir / "training_log.csv"

        # Logs outputs paths
        self.tb_dir = Path(config.get("tensorboard_dir", "./runs")) / model_name
        self.tb_writer = SummaryWriter(log_dir=str(self.tb_dir))

        # Mixed precision setup
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)

        # Multi-GPU Compatibility wrapper
        if torch.cuda.device_count() > 1 and bool(config.get("multi_gpu", False)):
            logger.info(f"Wrapping model with DataParallel using {torch.cuda.device_count()} GPUs.")
            self.model = nn.DataParallel(self.model)

        self.model = self.model.to(device)

        # Progress tracking variables
        self.start_epoch = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.history: List[Dict[str, Any]] = []

    def train(self, resume: bool = False) -> Dict[str, Any]:
        """Runs the training loop.

        Args:
            resume (bool): Whether to resume training from the last checkpoint.

        Returns:
            Dict[str, Any]: History metrics of the best training validation epoch.
        """
        if resume:
            self._load_checkpoint(self.last_pth)

        logger.info(f"Starting training run for model '{self.model_name}' on {self.device}...")
        start_time = time.time()

        for epoch in range(self.start_epoch, self.epochs):
            # 1. Training Pass
            train_loss = self._train_one_epoch(epoch)
            
            # 2. Validation Pass
            val_loss, val_psnr, val_ssim = self._validate(epoch)

            # 3. Learning rate step
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Retrieve active learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Log parameters to TensorBoard
            self.tb_writer.add_scalar("Loss/Train", train_loss, epoch)
            self.tb_writer.add_scalar("Loss/Val", val_loss, epoch)
            self.tb_writer.add_scalar("Metrics/Val_PSNR", val_psnr, epoch)
            self.tb_writer.add_scalar("Metrics/Val_SSIM", val_ssim, epoch)
            self.tb_writer.add_scalar("Params/LR", current_lr, epoch)

            # Append epoch metrics to history
            epoch_data = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_psnr": val_psnr,
                "val_ssim": val_ssim,
                "lr": current_lr
            }
            self.history.append(epoch_data)

            # Log to CSV
            self._log_to_csv(epoch_data)

            logger.info(
                f"Epoch [{epoch}/{self.epochs}] - Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val PSNR: {val_psnr:.2f}dB | Val SSIM: {val_ssim:.4f}"
            )

            # 4. Checkpoint check
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint(self.best_pth, epoch, val_loss, epoch_data)
            else:
                self.patience_counter += 1

            self._save_checkpoint(self.last_pth, epoch, self.best_val_loss, epoch_data)

            # 5. Early Stopping check
            if self.patience_counter >= self.patience:
                logger.info(f"Early stopping triggered after {epoch} epochs due to validation plateau.")
                break

        training_duration = time.time() - start_time
        logger.info(f"Training run completed in {training_duration:.2f}s.")

        # Save configuration to checkpoints folder
        import yaml
        with open(self.checkpoint_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(self.config, f)

        # Build experiment tracking metadata JSON
        experiment_meta = {
            "model_name": self.model_name,
            "dataset_version": self.dataset_version,
            "git_commit": get_git_commit(),
            "random_seed": self.seed,
            "hyperparameters": {
                "learning_rate": self.config.get("learning_rate", 1e-4),
                "optimizer": self.config.get("optimizer", "Adam"),
                "batch_size": self.config.get("batch_size", 8),
                "epochs": self.epochs
            },
            "training_time_seconds": training_duration,
            "best_epoch": next((item["epoch"] for item in self.history if item["val_loss"] == self.best_val_loss), 0),
            "best_validation_loss": self.best_val_loss,
            "checkpoint_path": str(self.best_pth.absolute())
        }
        with open(self.checkpoint_dir / "experiment_metadata.json", "w", encoding="utf-8") as f:
            import json
            json.dump(experiment_meta, f, indent=4)

        self.tb_writer.close()
        return experiment_meta

    def _train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        loss_accumulator = 0.0

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()

            # Automatic Mixed Precision
            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)

            # Scale gradients and step
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            loss_accumulator += loss.item()

        return loss_accumulator / len(self.train_loader)

    def _validate(self, epoch: int) -> Tuple[float, float, float]:
        self.model.eval()
        loss_accumulator = 0.0
        psnr_accumulator = 0.0
        ssim_accumulator = 0.0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                loss_accumulator += loss.item()

                # Calculate evaluation metrics
                # Outputs are clamped to [0.0, 1.0] for standard metrics compliance
                outputs_clamped = torch.clamp(outputs, 0.0, 1.0)
                psnr_accumulator += calculate_psnr(outputs_clamped, targets)
                ssim_accumulator += calculate_ssim(outputs_clamped, targets)

        num_batches = len(self.val_loader)
        return (
            loss_accumulator / num_batches,
            psnr_accumulator / num_batches,
            ssim_accumulator / num_batches
        )

    def _save_checkpoint(self, path: Path, epoch: int, best_loss: float, metrics: Dict[str, Any]) -> None:
        # Retrieve actual unwrapped state dict if Multi-GPU DataParallel wrapper is active
        model_to_save = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict(),
            "best_val_loss": best_loss,
            "seed": self.seed,
            "git_commit": get_git_commit(),
            "dataset_version": self.dataset_version,
            "hyperparameters": self.config,
            "val_metrics": metrics
        }
        torch.save(checkpoint, path)

    def _load_checkpoint(self, path: Path) -> None:
        if not path.exists():
            logger.warning(f"No checkpoint resolved at '{path}'. Starting training from scratch.")
            return

        logger.info(f"Loading execution checkpoint states from '{path}'...")
        checkpoint = torch.load(path, map_location=self.device)
        
        model_to_load = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        model_to_load.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if checkpoint.get("scaler_state_dict"):
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        self.start_epoch = checkpoint["epoch"] + 1
        self.best_val_loss = checkpoint["best_val_loss"]
        logger.info(f"Checkpoint loaded successfully. Resuming training at epoch {self.start_epoch}.")

    def _log_to_csv(self, epoch_data: Dict[str, Any]) -> None:
        df = pd.DataFrame([epoch_data])
        if not self.csv_log_pth.exists():
            df.to_csv(self.csv_log_pth, index=False)
        else:
            df.to_csv(self.csv_log_pth, mode="a", header=False, index=False)
