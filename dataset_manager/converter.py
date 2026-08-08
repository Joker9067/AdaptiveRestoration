"""
Converter module for the Semiconductor Image Restoration System.
Manages image format standardization (converting to grayscale) and slicing of 3D TIFF stacks.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Union
import cv2
import numpy as np
from PIL import Image, ImageSequence

logger = logging.getLogger(__name__)

class ImageConverter:
    """Handles image format standardizations, channel reductions, and TIFF stack slicing."""

    @staticmethod
    def load_as_grayscale(image_path: Path) -> np.ndarray:
        """Loads any image from disk and converts it to a single-channel grayscale image.

        Processes formats: PNG, JPG, JPEG, TIFF, BMP.
        If the image has color channels, it converts them to grayscale.

        Args:
            image_path (Path): Path to the image file.

        Returns:
            np.ndarray: Grayscale image as a 2D numpy array (uint8).

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is not a readable/valid image.
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Read the image using OpenCV. We read it as-is first (including color/alpha if present).
        # We don't read as grayscaled directly because we want to detect color channel configurations
        # for logging and precise conversions.
        try:
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                # Fallback to PIL in case OpenCV fails to read the format (e.g. some complex TIFFs)
                with Image.open(image_path) as PIL_img:
                    image_np = np.array(PIL_img)
                    image = image_np
        except Exception as e:
            raise ValueError(f"Failed to read image at {image_path}: {e}")

        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError(f"File at {image_path} is not a valid image or is empty.")

        # Determine number of dimensions
        ndim = image.ndim
        if ndim == 2:
            # Already grayscale
            return image
        
        if ndim == 3:
            channels = image.shape[2]
            if channels == 1:
                return np.squeeze(image, axis=2)
            elif channels == 3:
                # Convert BGR to Grayscale (OpenCV default is BGR, PIL is RGB)
                # Let's detect if we used PIL fallback or OpenCV
                # If we read it using standard cv2.imread, it returns BGR
                # We assume standard BGR for cv2.imread and RGB for PIL fallback
                # In order to be safe, let's use the ITU-R 601-2 standard: Y = 0.299R + 0.587G + 0.114B
                # Let's use OpenCV conversion.
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif channels == 4:
                # Convert BGRA to Grayscale
                return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            else:
                # Custom multi-channel handling (e.g. hyperspectral or multispectral electron detectors)
                # Compute average across all channels
                logger.warning(
                    f"Image {image_path.name} has unexpected number of channels ({channels}). Averaging channels."
                )
                return np.mean(image, axis=2).astype(np.uint8)

        raise ValueError(f"Unsupported image dimensions: {image.shape} for file {image_path}")

    @staticmethod
    def is_tiff_stack(image_path: Path) -> bool:
        """Determines if a TIFF file is a multi-page/3D stack.

        Args:
            image_path (Path): Path to the image.

        Returns:
            bool: True if it is a multi-page TIFF file, False otherwise.
        """
        if not image_path.exists():
            return False
        
        # Check matching file extensions
        ext = image_path.suffix.lower()
        if ext not in [".tif", ".tiff"]:
            return False

        try:
            with Image.open(image_path) as img:
                # A stack will have more than 1 frame
                return getattr(img, "is_animated", False) or img.n_frames > 1
        except Exception:
            return False

    @classmethod
    def slice_tiff_stack(
        cls, stack_path: Path, output_dir: Path
    ) -> List[Path]:
        """Slices a 3D TIFF stack into individual 2D grayscale PNG files.

        Saves them as output_dir / {stack_name}_slice_{idx:04d}.png.

        Args:
            stack_path (Path): Path to the multi-page TIFF stack.
            output_dir (Path): Destination folder for the sliced PNG files.

        Returns:
            List[Path]: Paths of the generated 2D PNG images.

        Raises:
            FileNotFoundError: If the stack file is missing.
            ValueError: If the file is not a valid TIFF stack.
        """
        if not stack_path.exists():
            raise FileNotFoundError(f"TIFF stack not found: {stack_path}")

        output_dir.mkdir(parents=True, exist_ok=True)
        stack_name = stack_path.stem
        saved_paths: List[Path] = []

        try:
            with Image.open(stack_path) as img:
                logger.info(f"Slicing TIFF stack: {stack_path.name} ({img.n_frames} frames)")
                
                for idx, frame in enumerate(ImageSequence.Iterator(img)):
                    # Convert PIL frame to numpy array
                    frame_np = np.array(frame)
                    
                    # Convert to grayscale standard uint8
                    if frame_np.ndim == 3:
                        # BGR to grayscale
                        if frame_np.shape[2] == 3:
                            frame_gray = cv2.cvtColor(frame_np, cv2.COLOR_RGB2GRAY)
                        elif frame_np.shape[2] == 4:
                            frame_gray = cv2.cvtColor(frame_np, cv2.COLOR_RGBA2GRAY)
                        else:
                            frame_gray = np.mean(frame_np, axis=2).astype(np.uint8)
                    else:
                        # Standardize to uint8 (TIFF stacks can be uint16 or float)
                        if frame_np.dtype != np.uint8:
                            # Normalize range to 0-255
                            f_min, f_max = frame_np.min(), frame_np.max()
                            if f_max > f_min:
                                frame_gray = ((frame_np - f_min) / (f_max - f_min) * 255.0).astype(np.uint8)
                            else:
                                frame_gray = np.zeros(frame_np.shape, dtype=np.uint8)
                        else:
                            frame_gray = frame_np

                    # Define target PNG output file path
                    slice_name = f"{stack_name}_slice_{idx:04d}.png"
                    slice_path = output_dir / slice_name
                    
                    # Save as PNG
                    success = cv2.imwrite(str(slice_path), frame_gray)
                    if not success:
                        # Fallback using PIL
                        Image.fromarray(frame_gray).save(slice_path)

                    saved_paths.append(slice_path)
                    logger.debug(f"Saved stack slice: {slice_path.name}")
                    
            return saved_paths

        except Exception as e:
            raise ValueError(f"Failed to slice TIFF stack {stack_path}: {e}")
