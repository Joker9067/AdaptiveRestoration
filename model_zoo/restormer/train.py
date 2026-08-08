"""
Training wrapper script for Restormer.
Sets up the dataset dataloaders, instantiates the model, and launches ModelTrainer.
"""
import os
import random
import logging
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset_manager.config import PipelineConfig
from dataset_manager.manager import DatasetManager
from model_zoo.restormer.model import Restormer
from model_zoo.restormer.config import get_config
from model_zoo.common.losses import CombinedWeightedLoss
from model_zoo.common.trainer import ModelTrainer

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s [%(name)s:%(lineno)d] - %(message)s")
logger = logging.getLogger(__name__)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main() -> None:
    config = get_config()
    seed = config.get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using execution device: {device}")

    workspace_path = Path(".").resolve()
    pipeline_cfg_path = workspace_path / "config.yaml"
    
    if not pipeline_cfg_path.exists():
        raise FileNotFoundError(f"Global configuration yaml not found at {pipeline_cfg_path}")
        
    pipeline_cfg = PipelineConfig.load_from_yaml(pipeline_cfg_path)
    
    manager = DatasetManager(pipeline_cfg)
    
    if not (Path(pipeline_cfg.paths.processed_dir) / "metadata.csv").exists():
        manager.prepare_and_process_all()

    train_set = manager.load(dataset_name="Synthetic_Generated", split="train", is_training=True)
    val_set = manager.load(dataset_name="Synthetic_Generated", split="val", is_training=False)

    batch_size = config.get("batch_size", 4)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    model = Restormer()

    opt_name = config.get("optimizer", "Adam").lower()
    lr = config.get("learning_rate", 1e-3)
    if opt_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

    sched_name = config.get("lr_scheduler", "CosineAnnealingLR").lower()
    epochs = config.get("epochs", 5)
    if sched_name == "cosineannealinglr":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif sched_name == "multisteplr":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epochs // 2, int(epochs * 0.8)])
    else:
        scheduler = None

    loss_fn = CombinedWeightedLoss(config.get("loss_weights", {"l1": 1.0}))

    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        config=config,
        device=device,
        model_name="restormer",
        dataset_version=pipeline_cfg.datasets.get("synthetic_generated", {}).get("version", "1.0"),
        seed=seed
    )
    trainer.train()

if __name__ == "__main__":
    main()
