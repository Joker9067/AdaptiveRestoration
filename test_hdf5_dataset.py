import os
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from dataset_manager.dataset import SemiconductorDataset
from dataset_manager.config import PipelineConfig
from main_pipeline import create_dataloaders
from dataset_audit import DatasetAuditor
import torch

def create_mock_h5():
    file_path = "dataset_20210407_denoising_ZIM504.h5"
    if os.path.exists(file_path):
        os.remove(file_path)
    with h5py.File(file_path, "w") as f:
        f.create_group("noisy")
        f.create_group("clean")
        
        # Create a mock 25x512x512 uint16 noisy volume
        noisy_data = np.random.randint(0, 65535, (25, 512, 512), dtype=np.uint16)
        f["noisy"].create_dataset("img_27", data=noisy_data)
        
        clean_data = np.random.randint(0, 65535, (25, 512, 512), dtype=np.uint16)
        f["clean"].create_dataset("img_27", data=clean_data)
    return file_path

def test_hdf5_volume():
    print("Setting up mock volumetric HDF5 environment...")
    h5_path = create_mock_h5()
    
    # Create mock dataframe replicating discovery logic
    records = []
    for i in range(25):
        records.append({
            "dataset_name": "mock_dataset",
            "dataset_type": "HDF5",
            "split": "train",
            "input_path": "dataset_20210407_denoising_ZIM504.h5::noisy/img_27",
            "ground_truth_path": "dataset_20210407_denoising_ZIM504.h5::clean/img_27",
            "frame_index": i,
            "group_hash": "parent_vol_hash",
            "clean_hash": f"frame_hash_{i}",
            "noisy_hash": f"n_frame_hash_{i}"
        })
    df = pd.DataFrame(records)
    
    dataset = SemiconductorDataset(df, base_dir=Path("."))
    
    print("Testing frame_index=0 ...")
    inp0, gt0 = dataset[0]
    assert inp0.shape == (1, 128, 128), f"Expected (1, 128, 128), got {inp0.shape}"
    
    print("Testing frame_index=24 ...")
    inp24, gt24 = dataset[24]
    assert inp24.shape == (1, 128, 128), f"Expected (1, 128, 128), got {inp24.shape}"
    
    print("Testing split consistency (mocking auditor)...")
    auditor = DatasetAuditor()
    auditor.records = records.copy()
    
    # We clear the manual splits to test the group_aware_split logic
    for r in auditor.records:
        r["split"] = ""
        
    auditor.group_aware_split()
    splits = set([r["split"] for r in auditor.records])
    assert len(splits) == 1, "Frames from the same volume ended up in different splits!"
    assigned_split = list(splits)[0]
    print(f"All 25 frames correctly assigned to identical split: {assigned_split}")
    
    # Check detecting leakage
    auditor.detect_leakage()
    assert auditor.stats["cross_split_leakage"] == 0, "Leakage detected!"
    assert auditor.stats["duplicate_samples"] == 0, "Frames incorrectly flagged as duplicates!"
    
    # Cleanup
    if os.path.exists(h5_path):
        os.remove(h5_path)
        
    print("VOLUMETRIC HDF5 TESTS: PASSED")

if __name__ == "__main__":
    test_hdf5_volume()
