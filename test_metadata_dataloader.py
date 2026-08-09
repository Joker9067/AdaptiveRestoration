import pandas as pd
from pathlib import Path
import torch
from dataset_manager.config import PipelineConfig
from main_pipeline import create_dataloaders

def test():
    meta_path = Path("unified_metadata.csv")
    df = pd.read_csv(meta_path, dtype={"input_path": str, "ground_truth_path": str, "split": str})
    
    inv_input = df['input_path'].isnull().sum() + (df['input_path'] == "").sum()
    inv_gt = df['ground_truth_path'].isnull().sum() + (df['ground_truth_path'] == "").sum()
    inv_split = df['split'].isnull().sum() + (df['split'] == "").sum()
    
    print("\nMETADATA VALIDATION: PASSED")
    print(f"Total records: {len(df)}")
    print(f"Invalid input paths: {inv_input}")
    print(f"Invalid target paths: {inv_gt}")
    print(f"Invalid splits: {inv_split}")
    
    pipeline_cfg = PipelineConfig.load_from_yaml(Path("config.yaml"))
    pipeline_cfg.dataloader.num_workers = 2
    train_loader, val_loader, test_loader = create_dataloaders(meta_path, pipeline_cfg, base_dir=Path("."))
    
    ds = train_loader.dataset
    sample_in, sample_gt = ds[0]
    print("FIRST DATASET SAMPLE: PASSED")
    print(f"Input shape: {sample_in.shape}, GT shape: {sample_gt.shape}")
    
    batch_in, batch_gt = next(iter(train_loader))
    print("FIRST DATALOADER BATCH: PASSED")
    print(f"Batch Input: {batch_in.shape}, Batch GT: {batch_gt.shape}")

if __name__ == "__main__":
    test()
