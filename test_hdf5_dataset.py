import os
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from dataset_manager.dataset import SemiconductorDataset
import torch

def create_mock_h5():
    file_path = "dataset.h5"
    with h5py.File(file_path, "w") as f:
        # Create groups
        f.create_group("noisy_1")
        f.create_group("clean")
        
        # Create a mock 128x128 uint16 noisy image
        noisy_data = np.random.randint(0, 65535, (128, 128), dtype=np.uint16)
        f["noisy_1"].create_dataset("img_41", data=noisy_data)
        
        # Create a mock clean image
        clean_data = np.random.randint(0, 65535, (128, 128), dtype=np.uint16)
        f["clean"].create_dataset("img_41", data=clean_data)
        
    return file_path

def test_hdf5_loader():
    print("Setting up mock HDF5 environment...")
    h5_path = create_mock_h5()
    
    # Create mock dataframe
    df = pd.DataFrame([{
        "dataset_name": "mock_dataset",
        "dataset_type": "HDF5",
        "split": "train",
        "input_path": "dataset.h5::noisy_1/img_41",
        "ground_truth_path": "dataset.h5::clean/img_41",
    }])
    
    dataset = SemiconductorDataset(df, base_dir=Path("."))
    
    print("Testing ds[0] lazy load...")
    inp, gt = dataset[0]
    
    assert inp.shape == (1, 128, 128), f"Expected input shape (1, 128, 128), got {inp.shape}"
    assert gt.shape == (1, 128, 128), f"Expected gt shape (1, 128, 128), got {gt.shape}"
    assert torch.max(inp) <= 1.0, "Input tensor not properly normalized to [0, 1]"
    
    print("FIRST DATASET SAMPLE: PASSED")
    print("HDF5 LAZY LOAD: PASSED")
    
    # Cleanup
    if os.path.exists(h5_path):
        os.remove(h5_path)

if __name__ == "__main__":
    test_hdf5_loader()
