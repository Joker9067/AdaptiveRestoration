"""
Downloader module for the Semiconductor Image Dataset System.
Implements SOLID downloader classes for:
1. Google Drive (via gdown)
2. Hugging Face (via huggingface_hub)
3. Zenodo (via requests Zenodo REST API)
4. Kaggle (via kaggle API with credentials fallback checks)
5. Local directories copy helper
"""

import os
import shutil
import logging
import zipfile
import tarfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
import requests

logger = logging.getLogger(__name__)

class BaseDownloader(ABC):
    """Abstract base class of SOLID downloaders."""

    @abstractmethod
    def download(self, target_dir: Path) -> Path:
        """Downloads the configured dataset files to the target directory.

        Args:
            target_dir (Path): Destination folder.

        Returns:
            Path: Path to the downloaded file or directory.
        """
        pass


class ArchiveExtractor:
    """Helper utility class to extract zip/tar artifacts and purge compressed archives."""

    @staticmethod
    def extract(archive_path: Path, extract_to: Path) -> None:
        """Determines archive format, extracts it, and deletes the source archive file.

        Args:
            archive_path (Path): Path to zip/tar file.
            extract_to (Path): Target folder.
        """
        if not archive_path.exists():
            raise FileNotFoundError(f"Archive missing: {archive_path}")

        extract_to.mkdir(parents=True, exist_ok=True)
        suffix = archive_path.suffix.lower()

        logger.info(f"Extracting {archive_path.name} to {extract_to}...")

        try:
            if suffix == ".zip":
                with zipfile.ZipFile(archive_path, "r") as zip_ref:
                    zip_ref.extractall(extract_to)
            elif suffix in [".tar", ".gz", ".tgz", ".bz2", ".tbz"]:
                with tarfile.open(archive_path, "r:*") as tar_ref:
                    tar_ref.extractall(extract_to)
            else:
                logger.warning(f"Unrecognized file compression '{suffix}'. Skipping extraction.")
                return

            logger.info("Extraction completed successfully.")
            
            # Delete zip/tar after successful extraction
            archive_path.unlink()
            logger.debug(f"Cleaned up zip source: {archive_path.name}")
        except Exception as e:
            logger.error(f"Failed extracting {archive_path.name}: {e}")
            raise e


class DownloaderRegistry:
    """Registry class for remote downloaders to comply with Open-Closed Principle."""
    _registry = {}

    @classmethod
    def register(cls, source_type: str):
        """Decorator to register a downloader subclass for a source type."""
        def decorator(subclass):
            cls._registry[source_type.lower()] = subclass
            return subclass
        return decorator

    @classmethod
    def get_downloader(cls, source_type: str, dataset_cfg) -> BaseDownloader:
        """Resolves and instantiates the registered downloader class."""
        key = source_type.lower()
        if key not in cls._registry:
            raise ValueError(f"No downloader registered for source type '{source_type}'")
        
        downloader_cls = cls._registry[key]
        
        if key == "google_drive":
            if not dataset_cfg.file_id:
                raise ValueError(f"Google Drive source type selected for '{dataset_cfg.name}', but no file_id is provided.")
            return downloader_cls(file_id=dataset_cfg.file_id)
        elif key == "huggingface":
            if not dataset_cfg.repo_id:
                raise ValueError(f"Hugging Face source type selected for '{dataset_cfg.name}', but no repo_id is provided.")
            return downloader_cls(repo_id=dataset_cfg.repo_id)
        elif key == "zenodo":
            if not dataset_cfg.record_id:
                raise ValueError(f"Zenodo source type selected for '{dataset_cfg.name}', but no record_id is provided.")
            return downloader_cls(record_id=dataset_cfg.record_id, file_name=dataset_cfg.file_name)
        elif key == "kaggle":
            if not dataset_cfg.dataset_slug:
                raise ValueError(f"Kaggle source type selected for '{dataset_cfg.name}', but no dataset_slug is provided.")
            return downloader_cls(dataset_slug=dataset_cfg.dataset_slug)
        elif key == "local":
            # Pass local path for copying
            return downloader_cls(source_path=Path(dataset_cfg.local_path))
        else:
            raise ValueError(f"Unsupported downloader key: {key}")


@DownloaderRegistry.register("google_drive")
class GoogleDriveDownloader(BaseDownloader):
    """Downloads files from Google Drive using gdown."""

    def __init__(self, file_id: str):
        """Initializes with the target Google Drive file identifier.

        Args:
            file_id (str): Google Drive unique file ID.
        """
        self.file_id = file_id

    def download(self, target_dir: Path) -> Path:
        """Downloads from Google Drive.

        Args:
            target_dir (Path): Destination folder.

        Returns:
            Path: Path to the downloaded archive zip.
        """
        import gdown
        target_dir.mkdir(parents=True, exist_ok=True)
        # Store downloaded zip temporarily within the target directory
        output_zip = target_dir / f"gdrive_{self.file_id}.zip"

        logger.info(f"Downloading Google Drive ID: {self.file_id} to {output_zip}")
        try:
            gdown.download(id=self.file_id, output=str(output_zip), quiet=False)
            if not output_zip.exists() or output_zip.stat().st_size < 100:
                # Try downloading using public direct link method if standard api fails
                url = f"https://drive.google.com/uc?id={self.file_id}"
                gdown.download(url, str(output_zip), quiet=False)
            
            return output_zip
        except Exception as e:
            logger.error(f"Google Drive download failed: {e}")
            raise e


@DownloaderRegistry.register("huggingface")
class HuggingFaceDownloader(BaseDownloader):
    """Downloads dataset snapshots or files from HuggingFace Hub."""

    def __init__(self, repo_id: str):
        """Initializes with HuggingFace Repository ID.

        Args:
            repo_id (str): HuggingFace dataset repo id e.g. 'mira-datasets/sem-images'.
        """
        self.repo_id = repo_id

    def download(self, target_dir: Path) -> Path:
        """Downloads/snapshots HF repo files directly to target path.

        Args:
            target_dir (Path): Output folder.

        Returns:
            Path: Root directory containing downloading data.
        """
        from huggingface_hub import snapshot_download
        logger.info(f"Downloading HuggingFace Repository: {self.repo_id}...")
        try:
            # Snapshot downloads handles folders tree directly
            snapshot_download(
                repo_id=self.repo_id,
                repo_type="dataset",
                local_dir=target_dir,
                local_dir_use_symlinks=False,
            )
            logger.info("HuggingFace download complete.")
            return target_dir
        except Exception as e:
            logger.error(f"HuggingFace snapshot download failed: {e}")
            raise e


@DownloaderRegistry.register("zenodo")
class ZenodoDownloader(BaseDownloader):
    """Downloads datasets from Zenodo standard REST repository API."""

    def __init__(self, record_id: str, file_name: Optional[str] = None):
        """Initializes with Record ID and filename details.

        Args:
            record_id (str): Zenodo record ID.
            file_name (str, optional): Target filename. If None/missing, downloads the first zip/tar it finds.
        """
        self.record_id = record_id
        self.file_name = file_name

    def download(self, target_dir: Path) -> Path:
        """Downloads files by searching Zenodo JSON records metadata.

        Args:
            target_dir (Path): Output folder.

        Returns:
            Path: Path to downloaded archive.
        """
        target_dir.mkdir(parents=True, exist_ok=True)
        api_url = f"https://zenodo.org/api/records/{self.record_id}"
        
        logger.info(f"Querying Zenodo API for record: {self.record_id}")
        try:
            response = requests.get(api_url, timeout=20)
            response.raise_for_status()
            data = response.json()
            
            files = data.get("files", [])
            if not files:
                raise ValueError(f"No files associated with Zenodo record: {self.record_id}")

            # Pick target file
            target_file_meta = None
            if self.file_name:
                for f in files:
                    if f.get("key") == self.file_name:
                        target_file_meta = f
                        break
            if not target_file_meta:
                # Default fallback: pick first zip/tar or first file listed
                for f in files:
                    key = f.get("key", "").lower()
                    if any(key.endswith(ext) for ext in [".zip", ".tar", ".gz", ".tgz"]):
                        target_file_meta = f
                        break
                if not target_file_meta:
                    target_file_meta = files[0]

            download_url = target_file_meta.get("links", {}).get("self")
            output_name = target_file_meta.get("key", f"zenodo_{self.record_id}.zip")
            output_path = target_dir / output_name

            if not download_url:
                raise ValueError("Could not resolve Zenodo file download URL link.")

            logger.info(f"Downloading from Zenodo: {download_url} -> {output_path}")
            
            # Stream download to target file
            with requests.get(download_url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(output_path, "wb") as f:
                    shutil.copyfileobj(r.raw, f)

            logger.info("Zenodo download finished.")
            return output_path
        except Exception as e:
            logger.error(f"Zenodo download failed: {e}")
            raise e


@DownloaderRegistry.register("kaggle")
class KaggleDownloader(BaseDownloader):
    """Downloads datasets from Kaggle metrics using the Kaggle API."""

    def __init__(self, dataset_slug: str):
        """Initializes with the Kaggle dataset slug (user/dataset).

        Args:
            dataset_slug (str): Example: 'kla-hackathon/sem-datasets'.
        """
        self.dataset_slug = dataset_slug

    def download(self, target_dir: Path) -> Path:
        """Downloads kaggle files using API credentials.

        If keys are missing, raises a descriptive error explaining environment variables setup.

        Args:
            target_dir (Path): Output folder.

        Returns:
            Path: Path to the target archive zip or directory.
        """
        logger.info(f"Initiating Kaggle Download for: {self.dataset_slug}...")
        
        # Verify Kaggle keys setup in environment or defaults
        kaggle_json = Path("~/.kaggle/kaggle.json").expanduser()
        if not kaggle_json.exists() and "KAGGLE_USERNAME" not in os.environ:
            msg = (
                "Kaggle API credentials not found! Please configure kaggle api key file in ~/.kaggle/kaggle.json "
                "or export environment variables KAGGLE_USERNAME and KAGGLE_KEY before proceeding."
            )
            logger.error(msg)
            raise RuntimeError(msg)

        try:
            # We import kaggle locally so it reads environment variables dynamically
            import kaggle
            target_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Downloading Kaggle dataset '{self.dataset_slug}' directly to {target_dir}...")
            # Downloads and extracts files automatically
            kaggle.api.dataset_download_files(self.dataset_slug, path=str(target_dir), unzip=True)
            
            logger.info("Kaggle download and auto-unzip complete.")
            return target_dir
        except Exception as e:
            logger.error(f"Kaggle API download failed: {e}")
            raise e


@DownloaderRegistry.register("local")
class LocalFolderDownloader(BaseDownloader):
    """Alternative downloader that copies or link folders locally if paths are specified."""

    def __init__(self, source_path: Path):
        """Initializes with local directory source.

        Args:
            source_path (Path): Path to source file registry.
        """
        self.source_path = Path(source_path)

    def download(self, target_dir: Path) -> Path:
        """Copies content from source path to destination path.

        Args:
            target_dir (Path): Destination path.

        Returns:
            Path: Target folder directory.
        """
        if not self.source_path.exists():
            raise FileNotFoundError(f"Source folder does not exist: {self.source_path}")

        logger.info(f"Copying files locally: {self.source_path} -> {target_dir}")
        if target_dir.exists():
            shutil.rmtree(target_dir)

        if self.source_path.is_file():
            target_dir.mkdir(parents=True, exist_ok=True)
            output_file = target_dir / self.source_path.name
            shutil.copy2(self.source_path, output_file)
            return output_file
        else:
            shutil.copytree(self.source_path, target_dir)
            return target_dir
