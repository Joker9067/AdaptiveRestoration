"""
Phase 1.5: Pre-Training Dataset Auditor for Semiconductor Image Restoration.
Scans, validates, deduplicates (SHA256), and categorizes all datasets.
"""

import os
import cv2
import json
import hashlib
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import h5py
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class DatasetAuditor:
    def __init__(self):
        self.search_dirs = [
            Path("/kaggle/input/datasets"),
            Path("/kaggle/input"),
            Path("./datasets"),
            Path(".")
        ]
        
        self.known_dirs = ["SEM", "NIST", "SBEM2-Z50-fast", "SBEM2-Z50-slow", "Synthetic_Generated"]
        for d in self.known_dirs:
            if Path(d).exists():
                self.search_dirs.append(Path(d).resolve())
                
        self.reports_dir = Path("reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        self.records = []
        self.visited_paths = set()
        self.hash_registry = {} # sha256 -> list of records
        
        self.stats = {
            "datasets_discovered": 0,
            "total_files": 0,
            "valid_image_samples": 0,
            "valid_supervised_pairs": 0,
            "rejected_samples": 0,
            "duplicate_samples": 0,
            "cross_split_leakage": 0,
            "leakage_train_train": 0,
            "leakage_train_val": 0,
            "leakage_train_test": 0,
            "leakage_val_test": 0,
            "leakage_cross_dataset": 0,
            "critical_errors": 0,
            "info_messages": [],
            "warning_messages": [],
            "critical_messages": []
        }
        
    def _hash_array(self, arr: np.ndarray) -> str:
        """Returns SHA256 hash of a numpy array payload."""
        return hashlib.sha256(arr.tobytes()).hexdigest()
        
    def _categorize_task(self, dataset_name: str) -> str:
        dataset_name = dataset_name.lower()
        if "sidd" in dataset_name or "fluorescence" in dataset_name or "synthetic" in dataset_name:
            return "Noise"
        elif "segmentation" in dataset_name or "3d" in dataset_name:
            return "Structure"
        elif "sbem" in dataset_name or "sem" in dataset_name:
            return "Super Resolution"
        return "Analyzer"

    def scan_standard_images(self, base_dir: Path):
        valid_exts = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}
        clean_dirs = [d for d in base_dir.rglob("*") if d.is_dir() and d.name.lower() in ["clean", "ground_truth", "gt", "target"]]
        
        for c_dir in clean_dirs:
            if 'processed' in c_dir.parts:
                continue
            parent = c_dir.parent
            noisy_dirs = [d for d in parent.iterdir() if d.is_dir() and d.name.lower() in ["noisy", "noisy_1", "noisy_2", "input", "raw"]]
            
            if not noisy_dirs:
                self.stats["rejected_samples"] += len(list(c_dir.iterdir()))
                self.stats["critical_messages"].append(f"Missing noisy counterpart for {c_dir}")
                self.stats["critical_errors"] += 1
                continue
                
            dataset_name = parent.name
            if dataset_name.lower() in ["train", "val", "test", "validation"]:
                dataset_name = parent.parent.name
                
            task_role = self._categorize_task(dataset_name)
            
            clean_files = {f.name: f for f in c_dir.iterdir() if f.suffix.lower() in valid_exts}
            for n_dir in noisy_dirs:
                noisy_files = {f.name: f for f in n_dir.iterdir() if f.suffix.lower() in valid_exts}
                common = set(clean_files.keys()).intersection(set(noisy_files.keys()))
                
                self.stats["total_files"] += len(clean_files) + len(noisy_files)
                
                for name in common:
                    try:
                        clean_path = clean_files[name].resolve()
                        noisy_path = noisy_files[name].resolve()
                        
                        if str(clean_path) in self.visited_paths:
                            continue
                        self.visited_paths.add(str(clean_path))
                        
                        clean_img = cv2.imread(str(clean_path), cv2.IMREAD_ANYDEPTH)
                        noisy_img = cv2.imread(str(noisy_path), cv2.IMREAD_ANYDEPTH)
                        
                        if clean_img is None or noisy_img is None:
                            self.stats["rejected_samples"] += 1
                            continue
                            
                        clean_hash = self._hash_array(clean_img)
                        noisy_hash = self._hash_array(noisy_img)
                        
                        h, w = clean_img.shape[:2]
                        if (h, w) != noisy_img.shape[:2]:
                            self.stats["info_messages"].append(f"Resolution mismatch in pair {name} ({dataset_name})")
                            
                        rec = {
                            "dataset_name": dataset_name,
                            "source_path": str(parent),
                            "image_id": name,
                            "clean_path": str(clean_path),
                            "noisy_path": str(noisy_path),
                            "clean_hash": clean_hash,
                            "noisy_hash": noisy_hash,
                            "resolution": f"{w}x{h}",
                            "bit_depth": clean_img.dtype.name,
                            "channels": 1 if len(clean_img.shape) == 2 else clean_img.shape[2],
                            "format": clean_path.suffix,
                            "category": task_role,
                            "split": "" # To be assigned group-wise
                        }
                        
                        self.records.append(rec)
                        self.stats["valid_image_samples"] += 2
                        self.stats["valid_supervised_pairs"] += 1
                        
                    except Exception as e:
                        self.stats["rejected_samples"] += 1
                        self.stats["warning_messages"].append(f"Failed to process {name}: {e}")

    def scan_hdf5(self, base_dir: Path):
        for h5_file in base_dir.rglob("*.h5"):
            if 'processed' in h5_file.parts:
                continue
            h5_resolved = str(h5_file.resolve())
            if h5_resolved in self.visited_paths:
                continue
            self.visited_paths.add(h5_resolved)
            
            self.stats["total_files"] += 1
            try:
                with h5py.File(h5_file, 'r') as f:
                    groups = list(f.keys())
                    clean_groups = [g for g in groups if "clean" in g.lower() or "gt" in g.lower() or "ground_truth" in g.lower() or "target" in g.lower()]
                    noisy_groups = [g for g in groups if "noisy" in g.lower() or "raw" in g.lower() or "input" in g.lower()]
                    
                    if not clean_groups or not noisy_groups:
                        self.stats["rejected_samples"] += 1
                        self.stats["critical_messages"].append(f"HDF5 {h5_file.name} missing clean/noisy groups.")
                        self.stats["critical_errors"] += 1
                        continue
                        
                    dataset_name = h5_file.stem
                    task_role = self._categorize_task(dataset_name)
                    clean_g = clean_groups[0]
                    clean_keys = list(f[clean_g].keys())
                    
                    for noisy_g in noisy_groups:
                        noisy_keys = list(f[noisy_g].keys())
                        common = set(clean_keys).intersection(set(noisy_keys))
                        
                        for k in common:
                            try:
                                clean_arr = f[clean_g][k][()]
                                noisy_arr = f[noisy_g][k][()]
                                
                                clean_hash = self._hash_array(clean_arr)
                                noisy_hash = self._hash_array(noisy_arr)
                                
                                h, w = clean_arr.shape[:2]
                                
                                rec = {
                                    "dataset_name": dataset_name,
                                    "source_path": str(h5_file),
                                    "image_id": k,
                                    "clean_path": f"{h5_file}::{clean_g}/{k}",
                                    "noisy_path": f"{h5_file}::{noisy_g}/{k}",
                                    "clean_hash": clean_hash,
                                    "noisy_hash": noisy_hash,
                                    "resolution": f"{w}x{h}",
                                    "bit_depth": clean_arr.dtype.name,
                                    "channels": 1 if len(clean_arr.shape) == 2 else clean_arr.shape[2],
                                    "format": "HDF5",
                                    "category": task_role,
                                    "split": "" # To be assigned group-wise
                                }
                                
                                self.records.append(rec)
                                self.stats["valid_image_samples"] += 2
                                self.stats["valid_supervised_pairs"] += 1
                                
                            except Exception as e:
                                self.stats["rejected_samples"] += 1
            except Exception as e:
                self.stats["critical_messages"].append(f"Failed to open HDF5 {h5_file.name}: {e}")
                self.stats["critical_errors"] += 1

    def group_aware_split(self):
        """Assigns train/val/test splits at the clean-image group level deterministically."""
        rng = np.random.RandomState(42)
        
        # Group by clean_hash
        groups = {}
        for rec in self.records:
            chash = rec["clean_hash"]
            if chash not in groups:
                groups[chash] = []
            groups[chash].append(rec)
            
        # Assign split
        sorted_hashes = sorted(groups.keys())
        for chash in sorted_hashes:
            split = rng.choice(["train", "val", "test"], p=[0.8, 0.1, 0.1])
            for rec in groups[chash]:
                rec["split"] = split

    def detect_leakage(self):
        """Cross-checks SHA256 hashes for data leakage across splits."""
        for idx, rec in enumerate(self.records):
            chash = rec["clean_hash"]
            nhash = rec["noisy_hash"]
            split = rec["split"]
            ds_name = rec["dataset_name"]
            
            for h in (chash, nhash):
                if h not in self.hash_registry:
                    self.hash_registry[h] = []
                
                # Check for leakage
                for existing in self.hash_registry[h]:
                    ex_split = existing["split"]
                    ex_ds = existing["dataset_name"]
                    
                    if ex_split != split:
                        self.stats["cross_split_leakage"] += 1
                        self.stats["critical_errors"] += 1
                        
                        # Track specific split combinations
                        combo = tuple(sorted([ex_split, split]))
                        if combo == ("train", "val"): self.stats["leakage_train_val"] += 1
                        elif combo == ("test", "train"): self.stats["leakage_train_test"] += 1
                        elif combo == ("test", "val"): self.stats["leakage_val_test"] += 1
                        
                        self.stats["critical_messages"].append(
                            f"LEAKAGE DETECTED: {rec['image_id']} ({split}) matches {existing['image_id']} ({ex_split})"
                        )
                    elif existing["image_id"] != rec["image_id"]:
                        self.stats["duplicate_samples"] += 1
                        if split == "train":
                            self.stats["leakage_train_train"] += 1
                        self.stats["info_messages"].append(
                            f"Duplicate content within {split}: {rec['image_id']} and {existing['image_id']}"
                        )
                        
                    if ex_ds != ds_name:
                        self.stats["leakage_cross_dataset"] += 1
                
                self.hash_registry[h].append(rec)

    def generate_reports(self):
        df = pd.DataFrame(self.records)
        df.to_csv(self.reports_dir / "dataset_audit.csv", index=False)
        
        with open(self.reports_dir / "dataset_audit.json", "w") as f:
            json.dump({"stats": self.stats, "records_count": len(self.records)}, f, indent=4)
            
        html = f"""
        <html><body>
        <h1>Dataset Audit Report</h1>
        <h3>Statistics</h3>
        <ul>
            <li>Valid Pairs: {self.stats['valid_supervised_pairs']}</li>
            <li>Rejected: {self.stats['rejected_samples']}</li>
            <li>Leakages: {self.stats['cross_split_leakage']}</li>
            <li>Critical Errors: {self.stats['critical_errors']}</li>
        </ul>
        <h3>Dataset Distribution</h3>
        {df.groupby(['dataset_name', 'split']).size().to_frame('count').to_html()}
        </body></html>
        """
        with open(self.reports_dir / "dataset_audit.html", "w") as f:
            f.write(html)
            
        return df

    def print_summary(self, df: pd.DataFrame, stage: int):
        print("==================================================")
        print("DATASET AUDIT SUMMARY")
        print("==================================================")
        print(f"Datasets discovered: {len(df['dataset_name'].unique()) if not df.empty else 0}")
        print(f"Total files: {self.stats['total_files']}")
        print(f"Valid image samples: {self.stats['valid_image_samples']}")
        print(f"Valid supervised pairs: {self.stats['valid_supervised_pairs']}")
        print(f"Rejected samples: {self.stats['rejected_samples']}")
        print(f"Duplicate samples (total): {self.stats['duplicate_samples']}")
        print(f"  - Train/Train duplicates: {self.stats['leakage_train_train']}")
        print(f"  - Cross-dataset duplicates: {self.stats['leakage_cross_dataset']}")
        print(f"Cross-split leakage: {self.stats['cross_split_leakage']}")
        print(f"  - Train/Val leakage: {self.stats['leakage_train_val']}")
        print(f"  - Train/Test leakage: {self.stats['leakage_train_test']}")
        print(f"  - Val/Test leakage: {self.stats['leakage_val_test']}")
        print(f"Critical errors: {self.stats['critical_errors']}")
        
        print("\nDataset                         Pairs        %")
        print("------------------------------------------------------")
        if not df.empty:
            ds_counts = df['dataset_name'].value_counts()
            total_pairs = self.stats['valid_supervised_pairs']
            for ds, count in ds_counts.items():
                pct = (count / total_pairs) * 100 if total_pairs > 0 else 0
                print(f"{ds:<30}  {count:<10} {pct:.1f}%")
                if pct > 80:
                    self.stats["warning_messages"].append(f"Dataset {ds} dominates {pct:.1f}% of total data.")
                    print(f"  [WARNING] Dataset {ds} dominates {pct:.1f}% of data.")
        print("------------------------------------------------------")
        print(f"TOTAL                           {self.stats['valid_supervised_pairs']}")
        
        train_count = len(df[df['split']=='train']) if not df.empty else 0
        val_count = len(df[df['split']=='val']) if not df.empty else 0
        test_count = len(df[df['split']=='test']) if not df.empty else 0
        
        print(f"\nTrain pairs: {train_count}")
        print(f"Validation pairs: {val_count}")
        print(f"Test pairs: {test_count}")
        
        print("\nDataset Audit Status:")
        
        if stage >= 2 and (train_count == 0 or val_count == 0 or test_count == 0):
            print("AUDIT STATUS: INSUFFICIENT DATA FOR TRAINING")
            self.stats["critical_errors"] += 1
            print("FAILED")
        elif self.stats["critical_errors"] > 0 or self.stats["valid_supervised_pairs"] == 0:
            print("FAILED")
        else:
            print("PASSED")
        print("==================================================")

    def run(self, stage: int = 1):
        logger.info("Starting rigorous dataset audit...")
        for base in self.search_dirs:
            if base.exists():
                self.scan_standard_images(base)
                self.scan_hdf5(base)
                
        self.group_aware_split()
        self.detect_leakage()
        df = self.generate_reports()
        self.print_summary(df, stage)
        
        if self.stats["critical_errors"] == 0 and not df.empty:
            out_df = df.rename(columns={"category": "dataset_type"})
            out_df["severity"] = "medium"
            out_df["noise_type"] = "mixed"
            out_path = Path("unified_metadata.csv")
            
            from dataset_manager.metadata import MetadataManager
            meta_mgr = MetadataManager(out_path)
            meta_mgr.write_metadata(out_df.to_dict('records'))
            logger.info("Generated unified metadata for Stage 2.")
            
        return self.stats["critical_errors"] == 0

if __name__ == "__main__":
    auditor = DatasetAuditor()
    auditor.run(stage=1)
