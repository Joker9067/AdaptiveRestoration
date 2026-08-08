"""
Metadata module for the Semiconductor Image Dataset System.
Manages metadata CSV files detailing datasets, image specs, and split locations.
Provides helper methods for clean Panda dataframes parsing.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

logger = logging.getLogger(__name__)

class MetadataManager:
    """Manages metadata collections, database exports, and database reads for the dataset registry."""

    REQUIRED_COLUMNS = [
        "image_name",
        "dataset_name",
        "version",
        "width",
        "height",
        "image_type",
        "ground_truth_path",
        "input_path",
        "split",
        "noise_type",
        "noise_level",
        "blur_level",
        "resolution_loss",
        "entropy",
        "texture_score",
        "edge_density",
        "fingerprint",
    ]

    def __init__(self, csv_path: Path):
        """Initializes metadata manager with a specific CSV file registry.

        Args:
            csv_path (Path): Path to the target CSV file.
        """
        self.csv_path = csv_path

    def write_metadata(self, records: List[Dict[str, Any]]) -> None:
        """Writes preprocessed records to metadata.csv.

        Args:
            records (List[Dict[str, Any]]): Database rows of preprocessed files.
        """
        if not records:
            logger.warning("No records provided. Skipping metadata CSV generation.")
            return

        df = pd.DataFrame(records)
        
        # Enforce column order and completeness for core requirements
        for col in self.REQUIRED_COLUMNS:
            if col not in df.columns:
                df[col] = "" # Fallback default empty cell
                
        # Move required columns to the front
        ordered_cols = self.REQUIRED_COLUMNS + [
            c for c in df.columns if c not in self.REQUIRED_COLUMNS
        ]
        df = df[ordered_cols]

        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.csv_path, index=False)
        logger.info(f"Successfully generated dataset metadata record registry at {self.csv_path}")

    def load_metadata(self) -> pd.DataFrame:
        """Loads and returns the metadata from metadata.csv as a Pandas DataFrame.

        Returns:
            pd.DataFrame: Loaded dataset records.

        Raises:
            FileNotFoundError: If the metadata CSV is missing.
            ValueError: If the file is not a valid CSV or has broken columns names.
        """
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Metadata CSV file not found: {self.csv_path}")

        try:
            df = pd.read_csv(self.csv_path)
        except Exception as e:
            raise ValueError(f"Failed to read metadata CSV: {e}")

        # Validate database integrity
        missing = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Invalid metadata schema. Missing required columns: {missing}")

        return df

    def filter_split(self, split: str) -> pd.DataFrame:
        """Loads metadata and filters by a specific dataset split (train/val/test).

        Args:
            split (str): 'train', 'val', or 'test'.

        Returns:
            pd.DataFrame: Split dataset records.
        """
        df = self.load_metadata()
        filtered = df[df["split"].str.lower() == split.lower()]
        return filtered.reset_index(drop=True)

    def filter_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Loads metadata and filters by a specific dataset name (e.g. SEM, NIST).

        Args:
            dataset_name (str): Selector name.

        Returns:
            pd.DataFrame: Matching dataset records.
        """
        df = self.load_metadata()
        filtered = df[df["dataset_name"].str.lower() == dataset_name.lower()]
        return filtered.reset_index(drop=True)
