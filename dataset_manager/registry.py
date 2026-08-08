"""
Dataset Registry module for the Semiconductor Image Restoration System.
Establishes the DatasetRegistry class and BaseDatasetHandler to adhere to the
Open-Closed Principle (OCP). Enables automatic registry of dataset handlers
(SEM, NIST, SIDD, SBF-SEM, Synthetic) without hardcoding conditioning branches.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple, Type, Optional
from PIL import Image

from dataset_manager.converter import ImageConverter

logger = logging.getLogger(__name__)

class BaseDatasetHandler(ABC):
    """Abstract base class define dataset-specific layouts, slicing, and pairing logics."""

    VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}

    @abstractmethod
    def can_handle(self, dataset_path: Path) -> bool:
        """Determines if this handler matches target dataset directory layout.

        Args:
            dataset_path (Path): Path to raw dataset subdirectory.

        Returns:
            bool: True if this handler matches.
        """
        pass

    @abstractmethod
    def pair_images(self, dataset_path: Path) -> List[Tuple[Path, Path]]:
        """Finds and pairs noisy input images with clean ground truth images under the dataset.

        Args:
            dataset_path (Path): Root folder of the dataset.

        Returns:
            List[Tuple[Path, Path]]: List of paired (noisy, clean) file paths.
        """
        pass

    def unpack_resources(self, dataset_path: Path) -> None:
        """Optional hook to extract or unpack resources (e.g. TIFF volumes) before pairing.

        Args:
            dataset_path (Path): Dataset root.
        """
        pass


class DatasetRegistry:
    """Registry cache holding mapping of handler classes to dataset string keys."""
    
    _handlers: Dict[str, Type[BaseDatasetHandler]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a BaseDatasetHandler subclass with a unique key name.

        Args:
            name (str): The unique dataset type identifier key.
        """
        def decorator(subclass: Type[BaseDatasetHandler]):
            cls._handlers[name.upper()] = subclass
            return subclass
        return decorator

    @classmethod
    def get_handler(cls, name: str) -> BaseDatasetHandler:
        """Retrieves an instantiated handler instance matching the key name.

        Args:
            name (str): Dataset key name.

        Returns:
            BaseDatasetHandler: The matched dataset handler.
        """
        key = name.upper()
        if key in cls._handlers:
            return cls._handlers[key]()
        
        logger.warning(f"No specific handler registered for key '{name}'. Returning DefaultHandler.")
        return DefaultHandler()

    @classmethod
    def detect_and_resolve(cls, dataset_path: Path) -> Tuple[str, BaseDatasetHandler]:
        """Iterates registered handlers to find one capable of processing dataset_path folder.

        Args:
            dataset_path (Path): Path to dataset folder.

        Returns:
            Tuple[str, BaseDatasetHandler]: Matched handler key name and its instance.
        """
        # Try checking registry handlers
        for name, handler_cls in cls._handlers.items():
            handler = handler_cls()
            if handler.can_handle(dataset_path):
                return name, handler

        # Heuristic fallback based on folder spelling
        dir_name = dataset_path.name.upper()
        for registered_name in cls._handlers.keys():
            if registered_name in dir_name or dir_name in registered_name:
                return registered_name, cls._handlers[registered_name]()

        return "DEFAULT", DefaultHandler()


# ----------------------------------------------------------------------
# Registering Specific Handles
# ----------------------------------------------------------------------

@DatasetRegistry.register("SEM")
class SEMHandler(BaseDatasetHandler):
    """Handles standard SEM parallel folders (clean/noisy or input/ground_truth)."""

    def can_handle(self, dataset_path: Path) -> bool:
        name = dataset_path.name.upper()
        return "SEM" in name and not any(k in name for k in ["NIST", "SBF", "SBEM", "SYNTHETIC"])

    def pair_images(self, dataset_path: Path) -> List[Tuple[Path, Path]]:
        all_files = [f for f in dataset_path.glob("**/*") if f.is_file() and f.suffix.lower() in self.VALID_EXTENSIONS]
        
        inputs = [f for f in all_files if any(x in [p.lower() for p in f.parts] for x in ["noisy", "input", "low"])]
        gts = [f for f in all_files if any(x in [p.lower() for p in f.parts] for x in ["clean", "ground_truth", "gt", "ref", "high"])]
        
        if not inputs or not gts:
            return []

        pairs = []
        gt_dict = {f.name: f for f in gts}
        for inp in inputs:
            if inp.name in gt_dict:
                pairs.append((inp, gt_dict[inp.name]))
        return pairs


@DatasetRegistry.register("NIST")
class NISTHandler(BaseDatasetHandler):
    """Handles NIST dataset layout, utilizing prefix/suffix naming conventions in flat directory."""

    def can_handle(self, dataset_path: Path) -> bool:
        return "NIST" in dataset_path.name.upper()

    def pair_images(self, dataset_path: Path) -> List[Tuple[Path, Path]]:
        all_files = [f for f in dataset_path.glob("*") if f.is_file() and f.suffix.lower() in self.VALID_EXTENSIONS]
        
        inputs = [f for f in all_files if "noisy" in f.name.lower()]
        gts = [f for f in all_files if "clean" in f.name.lower() or "gt" in f.name.lower()]
        
        pairs = []
        gt_dict = {f.name.lower().replace("_clean", "").replace("_gt", ""): f for f in gts}
        
        for inp in inputs:
            key = inp.name.lower().replace("_noisy", "")
            if key in gt_dict:
                pairs.append((inp, gt_dict[key]))
        return pairs


@DatasetRegistry.register("SIDD")
class SIDDHandler(BaseDatasetHandler):
    """Handles parallel noisy and gt subdirectories commonly used in SIDD structures."""

    def can_handle(self, dataset_path: Path) -> bool:
        return "SIDD" in dataset_path.name.upper()

    def pair_images(self, dataset_path: Path) -> List[Tuple[Path, Path]]:
        all_files = [f for f in dataset_path.glob("**/*") if f.is_file() and f.suffix.lower() in self.VALID_EXTENSIONS]
        
        inputs = [f for f in all_files if "noisy" in [p.lower() for p in f.parts]]
        gts = [f for f in all_files if any(x in [p.lower() for p in f.parts] for x in ["gt", "ground_truth", "ref"])]
        
        pairs = []
        gt_dict = {f.name: f for f in gts}
        for inp in inputs:
            if inp.name in gt_dict:
                pairs.append((inp, gt_dict[inp.name]))
        return pairs


@DatasetRegistry.register("SBF_SEM")
class SBFSEMHandler(BaseDatasetHandler):
    """Handles SBF-SEM slice unpacking of multi-page TIFF stacks and correlates paired slices."""

    def can_handle(self, dataset_path: Path) -> bool:
        return any(k in dataset_path.name.upper() for k in ["SBF_SEM", "SBEM", "SBF-SEM"])

    def unpack_resources(self, dataset_path: Path) -> None:
        """Locates multi-page TIFF frames and slices them into 2D directory structures prior to pairing."""
        tiff_stacks = [
            f for f in dataset_path.glob("**/*")
            if f.is_file() and f.suffix.lower() in [".tif", ".tiff"]
            and ImageConverter.is_tiff_stack(f)
        ]
        for ts in tiff_stacks:
            slice_out_dir = ts.parent / f"{ts.stem}_unpacked"
            ImageConverter.slice_tiff_stack(ts, slice_out_dir)

    def pair_images(self, dataset_path: Path) -> List[Tuple[Path, Path]]:
        all_files = [f for f in dataset_path.glob("**/*") if f.is_file() and f.suffix.lower() in self.VALID_EXTENSIONS]
        
        # Sort files into input and gt based on parents or suffixes
        inputs = []
        gts = []
        
        for f in all_files:
            p_parts = [p.lower() for p in f.parts]
            # Match unpacked slices based on folder context
            if any(x in p_parts for x in ["noisy", "input"]):
                inputs.append(f)
            elif any(x in p_parts for x in ["clean", "gt", "ref", "ground_truth"]):
                gts.append(f)

        if not inputs or not gts:
            return []

        pairs = []
        gt_dict = {f.name: f for f in gts}
        for inp in inputs:
            if inp.name in gt_dict:
                pairs.append((inp, gt_dict[inp.name]))
        return pairs


@DatasetRegistry.register("SYNTHETIC")
class SyntheticHandler(BaseDatasetHandler):
    """Handles synthetic databases with flat paths and suffix indicators."""

    def can_handle(self, dataset_path: Path) -> bool:
        return "SYNTHETIC" in dataset_path.name.upper() or "SYN" in dataset_path.name.upper()

    def pair_images(self, dataset_path: Path) -> List[Tuple[Path, Path]]:
        all_files = [f for f in dataset_path.glob("*") if f.is_file() and f.suffix.lower() in self.VALID_EXTENSIONS]
        
        inputs = [f for f in all_files if "noisy" in f.name.lower()]
        gts = [f for f in all_files if "gt" in f.name.lower() or "clean" in f.name.lower()]
        
        pairs = []
        gt_dict = {f.name.lower().replace("_gt", "").replace("_clean", ""): f for f in gts}
        for inp in inputs:
            key = inp.name.lower().replace("_noisy", "")
            if key in gt_dict:
                pairs.append((inp, gt_dict[key]))
        return pairs


class DefaultHandler(BaseDatasetHandler):
    """Fallback handler that resolves sorting-based sequential alignments."""

    def can_handle(self, dataset_path: Path) -> bool:
        return True

    def pair_images(self, dataset_path: Path) -> List[Tuple[Path, Path]]:
        all_files = [f for f in dataset_path.glob("**/*") if f.is_file() and f.suffix.lower() in self.VALID_EXTENSIONS]
        if len(all_files) % 2 != 0 or len(all_files) < 2:
            return []
        
        # Sort and alternate
        sorted_files = sorted(all_files, key=lambda f: (str(f.parent), f.name))
        half = len(sorted_files) // 2
        sub_dirs = list({f.parent for f in sorted_files})
        
        if len(sub_dirs) == 2:
            sub0 = sorted([f for f in sorted_files if f.parent == sub_dirs[0]])
            sub1 = sorted([f for f in sorted_files if f.parent == sub_dirs[1]])
            if len(sub0) == len(sub1):
                # Guess which is input based on directory spelling
                s0_name = sub_dirs[0].name.lower()
                s1_name = sub_dirs[1].name.lower()
                s0_is_input = True
                if any(x in s1_name for x in ["noisy", "input", "low"]) or any(x in s0_name for x in ["clean", "gt", "high", "ref"]):
                    s0_is_input = False
                
                if s0_is_input:
                    return list(zip(sub0, sub1))
                else:
                    return list(zip(sub1, sub0))
                    
        # Sequential list splitting
        pairs = []
        for i in range(0, len(sorted_files), 2):
            pairs.append((sorted_files[i], sorted_files[i+1]))
        return pairs
