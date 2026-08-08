"""
Dataset Validator module for the Semiconductor Image Restoration System.
Verifies dataset integrity, matches pairs, checks channels/resolutions,
detects duplicate file content via SHA256 hashes, verifies filename conventions,
and computes an overall Dataset Quality Score (0-100).
"""

import logging
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Tuple
import cv2

logger = logging.getLogger(__name__)

class DatasetValidator:
    """Performs validation sweeps on preprocessed image folders and paired items."""

    def __init__(self, metadata_path: Path, target_size: Tuple[int, int]):
        """Initializes with metadata CSV and target specs.

        Args:
            metadata_path (Path): Path to metadata.csv registry.
            target_size (Tuple[int, int]): Target image height and width (H, W).
        """
        self.metadata_path = Path(metadata_path)
        self.target_size = target_size  # (H, W)

    @staticmethod
    def _calculate_file_sha256(file_path: Path) -> str:
        """Computes the SHA256 hash of a file on disk.

        Args:
            file_path (Path): Path to the target file.

        Returns:
            str: Hex digest of SHA256 hash.
        """
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(65536), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.warning(f"Failed to calculate SHA256 for {file_path}: {e}")
            return ""

    def validate(self) -> Dict[str, Any]:
        """Runs the validation checks over all registered pairs.

        Returns:
            Dict[str, Any]: The structured validation sweep report including Dataset Quality Score.
        """
        import pandas as pd
        
        report = {
            "metadata_exists": False,
            "total_records": 0,
            "broken_images": [],
            "missing_images": [],
            "channel_errors": [],
            "resolution_errors": [],
            "mismatched_pairs": [],
            "duplicate_images": [],
            "filename_errors": [],
            "valid_records_count": 0,
            "quality_score": 100,
            "status": "PASS",
        }

        if not self.metadata_path.exists():
            report["status"] = "FAIL"
            report["quality_score"] = 0
            report["error"] = f"Metadata file not found at {self.metadata_path}"
            logger.error(report["error"])
            return report

        report["metadata_exists"] = True
        try:
            df = pd.read_csv(self.metadata_path)
        except Exception as e:
            report["status"] = "FAIL"
            report["quality_score"] = 0
            report["error"] = f"Failed to load metadata CSV: {e}"
            logger.error(report["error"])
            return report

        report["total_records"] = len(df)
        base_dir = self.metadata_path.parent.parent  # Resolve processed parent folder
        
        valid_counter = 0
        file_hashes: Dict[str, List[Path]] = {}
        seen_filenames = set()

        for idx, row in df.iterrows():
            img_name = row.get("image_name", f"index_{idx}")
            inp_rel = row.get("input_path")
            gt_rel = row.get("ground_truth_path")
            dataset_name = row.get("dataset_name", "UNKNOWN")

            if not inp_rel or not gt_rel:
                msg = f"Record {idx} ({img_name}): Input or Ground Truth path missing in metadata registry."
                report["mismatched_pairs"].append(msg)
                logger.warning(msg)
                continue

            inp_path = base_dir.parent / Path(inp_rel)
            gt_path = base_dir.parent / Path(gt_rel)

            has_error = False

            # 1. Filename validation checks (no spaces, unique filenames)
            if " " in img_name:
                report["filename_errors"].append(f"Image filename contains spaces: {img_name}")
                has_error = True
            if img_name in seen_filenames:
                report["filename_errors"].append(f"Duplicate image filename in registry: {img_name}")
                has_error = True
            else:
                seen_filenames.add(img_name)

            # 2. Check file existence
            if not inp_path.exists():
                report["missing_images"].append(f"Noisy input file missing on disk: {inp_path}")
                has_error = True
            if not gt_path.exists():
                report["missing_images"].append(f"Ground truth file missing on disk: {gt_path}")
                has_error = True

            if has_error:
                continue

            # 3. Check loading integrity (broken images)
            try:
                inp_img = cv2.imread(str(inp_path), cv2.IMREAD_UNCHANGED)
                gt_img = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
            except Exception as e:
                report["broken_images"].append(f"Image load failure for {img_name}: {e}")
                continue

            if inp_img is None:
                report["broken_images"].append(f"Corrupted noisy image (fails to load): {inp_path}")
                has_error = True
            if gt_img is None:
                report["broken_images"].append(f"Corrupted ground truth image (fails to load): {gt_path}")
                has_error = True

            if has_error:
                continue

            # 4. Check wrong channels (Should be strictly grayscale single-channel)
            if len(inp_img.shape) != 2:
                report["channel_errors"].append(
                    f"Noisy image {img_name} is not grayscale. Channels count found: {inp_img.shape[2] if len(inp_img.shape) > 2 else 0}"
                )
                has_error = True
            if len(gt_img.shape) != 2:
                report["channel_errors"].append(
                    f"Ground truth image {img_name} is not grayscale. Channels count found: {gt_img.shape[2] if len(gt_img.shape) > 2 else 0}"
                )
                has_error = True

            # 5. Check wrong resolutions
            th, tw = self.target_size
            if inp_img.shape[:2] != (th, tw):
                report["resolution_errors"].append(
                    f"Noisy image {img_name} shape mismatch. Expected {(th, tw)}, got {inp_img.shape[:2]}"
                )
                has_error = True
            if gt_img.shape[:2] != (th, tw):
                report["resolution_errors"].append(
                    f"Ground truth image {img_name} shape mismatch. Expected {(th, tw)}, got {gt_img.shape[:2]}"
                )
                has_error = True

            # 6. Mismatched shapes between pair elements
            if inp_img.shape != gt_img.shape:
                report["mismatched_pairs"].append(
                    f"Pair size mismatch for {img_name}: Noisy is {inp_img.shape}, GT is {gt_img.shape}"
                )
                has_error = True

            # 7. Hash-based duplicate contents check
            inp_hash = self._calculate_file_sha256(inp_path)
            gt_hash = self._calculate_file_sha256(gt_path)

            for hash_val, file_path in [(inp_hash, inp_path), (gt_hash, gt_path)]:
                if hash_val:
                    if hash_val not in file_hashes:
                        file_hashes[hash_val] = []
                    file_hashes[hash_val].append(file_path)

            if not has_error:
                valid_counter += 1

        # Process duplicates from hash registry
        for hash_val, paths in file_hashes.items():
            if len(paths) > 1:
                rel_paths = [str(p.name) for p in paths]
                report["duplicate_images"].append(f"Hash duplicate group ({len(paths)} files): {', '.join(rel_paths)}")

        report["valid_records_count"] = valid_counter

        # Calculate Quality Score (0 - 100)
        total_deductions = 0
        total_deductions += min(20, len(report["missing_images"]) * 10)
        total_deductions += min(20, len(report["broken_images"]) * 10)
        total_deductions += min(20, len(report["channel_errors"]) * 5)
        total_deductions += min(20, len(report["resolution_errors"]) * 5)
        total_deductions += min(20, len(report["mismatched_pairs"]) * 5)
        total_deductions += min(10, len(report["duplicate_images"]) * 2)
        total_deductions += min(10, len(report["filename_errors"]) * 2)

        report["quality_score"] = max(0, 100 - total_deductions)

        # Resolve status
        critical_errors = len(report["broken_images"]) + len(report["missing_images"])
        if critical_errors > 0 or report["valid_records_count"] == 0:
            report["status"] = "FAIL"
        elif total_deductions > 0:
            report["status"] = "WARNING"
        else:
            report["status"] = "PASS"

        logger.info(
            f"Dataset Validation completed. Status: {report['status']}. Score: {report['quality_score']}/100. "
            f"Validated {valid_counter}/{report['total_records']} records successfully."
        )
        return report

    def save_validation_report(self, output_path: Path) -> None:
        """Saves the validation reports to a destination JSON file.

        Args:
            output_path (Path): Path to output json.
        """
        report = self.validate()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)
        logger.info(f"Saved validation audit report to {output_path.absolute()}")
