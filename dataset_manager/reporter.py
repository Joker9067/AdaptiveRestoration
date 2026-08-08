"""
Dataset Reporter module for the Semiconductor Image Restoration System.
Generates a publication-quality HTML Diagnostic & Audit report
containing global statistics, validation audits, histograms,
and paired image panels.
"""

import base64
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class DatasetReporter:
    """Combines metadata checks, validations, and statistics into a standalone, interactive HTML dashboard."""

    def __init__(self, metadata_path: Path):
        """Initializes with preprocessed metadata CSV file path.

        Args:
            metadata_path (Path): Path to metadata.csv.
        """
        self.metadata_path = Path(metadata_path)

    @staticmethod
    def _image_to_base64(image_path: Path) -> str:
        """Converts an image file on disk to a base64 string for direct HTML embedding."""
        if not image_path.exists():
            return ""
        try:
            with open(image_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
                ext = image_path.suffix.lower().replace(".", "")
                mime = f"image/{ext}" if ext != "jpg" else "image/jpeg"
                return f"data:{mime};base64,{encoded}"
        except Exception as e:
            logger.warning(f"Could not convert {image_path.name} to base64: {e}")
            return ""

    def generate_html_report(
        self,
        stats: Dict[str, Any],
        validation: Dict[str, Any],
        config_datasets: Dict[str, Any],
        output_html_path: Path
    ) -> None:
        """Assembles data components, selects representative pairs, and writes report to disk.

        Args:
            stats (Dict[str, Any]): Outputs from DatasetStatsGenerator.
            validation (Dict[str, Any]): Outputs from DatasetValidator.
            config_datasets (Dict[str, Any]): Dataset registries entries.
            output_html_path (Path): Output filename.
        """
        import pandas as pd
        logger.info(f"Compiling HTML Diagnostic Report at {output_html_path.absolute()}")
        
        # Load dataset profile
        profile_path = self.metadata_path.parent / "dataset_profile.json"
        profile_data = {}
        if profile_path.exists():
            try:
                with open(profile_path, "r", encoding="utf-8") as f:
                    profile_data = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load dataset profile for report: {e}")

        # Load metadata and select 3 representatives to embed as base64 samples
        sample_rows_html = ""
        base_dir = self.metadata_path.parent.parent
        
        if self.metadata_path.exists():
            try:
                df = pd.read_csv(self.metadata_path)
                # Group by dataset and pick 1 sample from each to show diversity
                sampled_groups = df.groupby("dataset_name").first().reset_index()
                
                for _, row in sampled_groups.head(4).iterrows():
                    ds_name = row.get("dataset_name", "UNKNOWN")
                    img_name = row.get("image_name", "UNKNOWN")
                    
                    inp_rel = row.get("input_path")
                    gt_rel = row.get("ground_truth_path")
                    
                    inp_path = base_dir.parent / Path(inp_rel)
                    gt_path = base_dir.parent / Path(gt_rel)
                    
                    inp_b64 = self._image_to_base64(inp_path)
                    gt_b64 = self._image_to_base64(gt_path)
                    
                    split = row.get("split", "train")
                    
                    # Fetch from profile if available, otherwise calculate fallback
                    psnr = 0.0
                    ssim = 0.0
                    noise_lvl = 0.0
                    blur_lvl = 0.0
                    entropy = 0.0
                    texture = 0.0
                    edge = 0.0
                    
                    if profile_data and "image_profiles" in profile_data:
                        img_profile = profile_data["image_profiles"].get(img_name, {})
                        if img_profile:
                            psnr = img_profile.get("psnr", 0.0)
                            ssim = img_profile.get("ssim", 0.0)
                            noise_lvl = img_profile.get("noise_level", 0.0)
                            blur_lvl = img_profile.get("blur_level", 0.0)
                            entropy = img_profile.get("noisy_entropy", 0.0)
                            texture = img_profile.get("noisy_texture_complexity", 0.0)
                            edge = img_profile.get("noisy_edge_density", 0.0)
                    
                    if psnr == 0.0:
                        # Fallback calculations
                        try:
                            inp_img = cv2.imread(str(inp_path), cv2.IMREAD_GRAYSCALE)
                            gt_img = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
                            if inp_img is not None and gt_img is not None:
                                mse = float(np.mean((inp_img.astype(np.float32) - gt_img.astype(np.float32))**2))
                                if mse > 0:
                                    psnr = float(20 * np.log10(255.0 / np.sqrt(mse)))
                        except Exception:
                            pass

                    sample_rows_html += f"""
                    <div class="card mb-4 bg-dark-card border-secondary text-light">
                        <div class="card-header d-flex justify-content-between align-items-center">
                            <span class="badge bg-primary text-light font-monospace">{ds_name}</span>
                            <span class="text-muted font-monospace">{img_name} ({split})</span>
                        </div>
                        <div class="card-body">
                            <div class="row align-items-center">
                                <div class="col-md-5 text-center">
                                    <div class="img-title font-monospace mb-2 text-secondary">Noisy Input</div>
                                    <div class="zoom-container">
                                        <img src="{inp_b64}" class="img-fluid rounded border border-light sample-img" alt="Input">
                                    </div>
                                </div>
                                <div class="col-md-2 text-center">
                                    <div class="metric-grid">
                                        <div class="metric-block py-2 px-1 mb-2">
                                            <div class="label font-monospace">PSNR</div>
                                            <div class="val font-monospace fs-5 text-info">{psnr:.2f} dB</div>
                                        </div>
                                        <div class="metric-block py-2 px-1 mb-2">
                                            <div class="label font-monospace">SSIM</div>
                                            <div class="val font-monospace fs-5 text-success">{ssim:.3f}</div>
                                        </div>
                                        <div class="metric-block py-2 px-1 mb-2">
                                            <div class="label font-monospace">Noise Std</div>
                                            <div class="val font-monospace fs-6 text-warning">{noise_lvl:.4f}</div>
                                        </div>
                                        <div class="metric-block py-2 px-1 mb-2">
                                            <div class="label font-monospace">Blur Score</div>
                                            <div class="val font-monospace fs-6 text-danger">{blur_lvl:.2f}</div>
                                        </div>
                                    </div>
                                </div>
                                <div class="col-md-5 text-center">
                                    <div class="img-title font-monospace mb-2 text-secondary">Ground Truth Target</div>
                                    <div class="zoom-container">
                                        <img src="{gt_b64}" class="img-fluid rounded border border-light sample-img" alt="Ground Truth">
                                    </div>
                                </div>
                            </div>
                            <div class="row mt-3 text-center border-top border-secondary pt-3">
                                <div class="col-4">
                                    <div class="text-secondary small font-monospace">SHANNON ENTROPY</div>
                                    <div class="text-light font-monospace fw-bold">{entropy:.2f} bits</div>
                                </div>
                                <div class="col-4">
                                    <div class="text-secondary small font-monospace">CANNY EDGE DENSITY</div>
                                    <div class="text-light font-monospace fw-bold">{edge:.4f}</div>
                                </div>
                                <div class="col-4">
                                    <div class="text-secondary small font-monospace">TEXTURE SCORE</div>
                                    <div class="text-light font-monospace fw-bold">{texture:.1f}</div>
                                </div>
                            </div>
                        </div>
                    </div>
                    """
            except Exception as e:
                logger.warning(f"Could not prepare sample preview panels: {e}")
                sample_rows_html = "<div class='alert alert-warning'>No previews could be resolved.</div>"
        else:
            sample_rows_html = "<div class='alert alert-warning'>Metadata CSV not found. Preview unavailable.</div>"

        # Expand dataset registry summaries list
        datasets_list_html = ""
        fingerprint_display = "N/A"
        for name, ds_cfg in config_datasets.items():
            version = getattr(ds_cfg, "version", "1.0")
            source_type = getattr(ds_cfg, "source_type", "local")
            sha256 = getattr(ds_cfg, "sha256", "N/A")
            sha_display = sha256[:10] + "..." if sha256 and sha256 != "N/A" else "N/A"
            
            # Match counts
            pairs_count = stats.get("datasets_summary", {}).get(name.upper(), {}).get("pairs_count", 0)

            datasets_list_html += f"""
            <tr>
                <td class="text-light font-monospace fw-bold">{name.upper()}</td>
                <td><span class="badge bg-secondary font-monospace">{version}</span></td>
                <td class="text-capitalize">{source_type}</td>
                <td><span class="text-muted font-monospace" title="{sha256}">{sha_display}</span></td>
                <td class="text-light text-end font-monospace">{pairs_count} pairs</td>
            </tr>
            """

        # Fetch Fingerprint from metadata if exists
        if self.metadata_path.exists():
            try:
                mdf = pd.read_csv(self.metadata_path)
                if "fingerprint" in mdf.columns and not mdf.empty:
                    fingerprint_display = str(mdf.iloc[0]["fingerprint"])
            except Exception:
                pass

        # Format audit checklist badge status
        val_status = validation.get("status", "PASS")
        val_score = validation.get("quality_score", 100)
        status_color = "success" if val_status == "PASS" else ("warning" if val_status == "WARNING" else "danger")
        
        # Build checklist summary
        total_failures = (
            len(validation.get("missing_images", [])) +
            len(validation.get("broken_images", [])) +
            len(validation.get("channel_errors", [])) +
            len(validation.get("resolution_errors", [])) +
            len(validation.get("mismatched_pairs", [])) +
            len(validation.get("duplicate_images", [])) +
            len(validation.get("filename_errors", []))
        )

        checked_items = [
            ("Image Files Existence Verification", len(validation.get("missing_images", [])) == 0),
            ("Loading Integrity (Corrupted files checks)", len(validation.get("broken_images", [])) == 0),
            ("Grayscale Color Check (Wrong channel audits)", len(validation.get("channel_errors", [])) == 0),
            ("Target Spatial Scale Matching (Resolution)", len(validation.get("resolution_errors", [])) == 0),
            ("Clean & Noisy Dimensional Synchronization", len(validation.get("mismatched_pairs", [])) == 0),
            ("Duplicate Files content checks (SHA256)", len(validation.get("duplicate_images", [])) == 0),
            ("Filename Convention checks (no spaces, duplicates)", len(validation.get("filename_errors", [])) == 0),
        ]

        check_list_html = ""
        for title, ok in checked_items:
            icon = "check-circle-fill text-success" if ok else "x-circle-fill text-danger"
            check_list_html += f"""
            <li class="list-group-item bg-dark-card border-secondary text-light d-flex justify-content-between align-items-center py-3">
                <span>{title}</span>
                <i class="bi bi-{icon} fs-5"></i>
            </li>
            """

        # Detailed validation errors details list
        errors_details_html = ""
        if total_failures > 0:
            errors_details_html += "<div class='text-danger fw-bold mb-2'>Audit Exception Log:</div>"
            all_errors = (
                validation.get("missing_images", []) +
                validation.get("broken_images", []) +
                validation.get("channel_errors", []) +
                validation.get("resolution_errors", []) +
                validation.get("mismatched_pairs", []) +
                validation.get("duplicate_images", []) +
                validation.get("filename_errors", [])
            )
            for err in all_errors[:20]: # Show up to 20 errors
                errors_details_html += f"<div class='text-warning border-bottom border-secondary py-1 font-monospace small'>- {err}</div>"
            if len(all_errors) > 20:
                errors_details_html += f"<div class='text-muted mt-2 font-monospace small'>... and {len(all_errors)-20} more errors. See validation_report.json</div>"
        else:
            errors_details_html = "<div class='text-success fw-bold font-monospace py-2'><i class='bi bi-shield-check me-2'></i>Integrity sweep returned zero warnings. Clean build.</div>"

        # Histograms data
        noisy_hist_list = stats.get("noisy_stats", {}).get("intensity_histogram_256", [0]*256)
        clean_hist_list = stats.get("clean_stats", {}).get("intensity_histogram_256", [0]*256)

        # Profile dataset metrics arrays
        profile_labels = []
        profile_psnr = []
        profile_ssim = []
        profile_noise = []
        profile_edge = []
        profile_texture = []

        if profile_data and "dataset_profiles" in profile_data:
            for ds_k, ds_p in profile_data["dataset_profiles"].items():
                profile_labels.append(ds_k)
                profile_psnr.append(ds_p.get("avg_psnr", 0.0))
                profile_ssim.append(ds_p.get("avg_ssim", 0.0))
                profile_noise.append(ds_p.get("avg_noise_level", 0.0))
                profile_edge.append(ds_p.get("avg_edge_density_noisy", 0.0))
                profile_texture.append(ds_p.get("avg_texture_complexity_noisy", 0.0))

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Semiconductor Dataset Diagnostics & Profiler Report</title>
    
    <!-- Premium UI Frameworks: Bootstrap Dark & Icons -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css" rel="stylesheet">
    
    <!-- Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
    
    <style>
        body {{
            background-color: #0b0f19;
            color: #f1f5f9;
            font-family: 'Inter', sans-serif;
            min-height: 100vh;
        }}
        .font-monospace {{
            font-family: 'JetBrains Mono', monospace !important;
        }}
        .bg-dark-card {{
            background-color: #111b2d !important;
            border-color: #1e293b !important;
        }}
        .glass {{
            background: rgba(17, 27, 45, 0.7);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.05);
        }}
        .card-title {{
            font-weight: 700;
            letter-spacing: -0.025em;
        }}
        .sample-img {{
            max-height: 300px;
            object-fit: contain;
            background: #020617;
            width: 100%;
            transition: transform 0.3s ease;
        }}
        .zoom-container {{
            overflow: hidden;
            border-radius: 6px;
            border: 1px solid #334155;
            background: #020617;
        }}
        .zoom-container:hover .sample-img {{
            transform: scale(1.15);
        }}
        .metric-block {{
            background: #080d16;
            border-radius: 6px;
            border: 1px solid #1e293b;
        }}
        .metric-block .label {{
            font-size: 0.7rem;
            text-transform: uppercase;
            color: #64748b;
        }}
        td, th {{
            border-color: #1e293b !important;
            vertical-align: middle;
        }}
        th {{
            color: #94a3b8 !important;
            font-size: 0.85rem;
            text-transform: uppercase;
        }}
        .list-group-item {{
            border-color: #1e293b !important;
        }}
        .nav-tabs {{
            border-bottom: 2px solid #1e293b;
        }}
        .nav-tabs .nav-link {{
            color: #94a3b8;
            border: none;
            font-weight: 500;
            padding: 12px 20px;
        }}
        .nav-tabs .nav-link.active {{
            background-color: transparent;
            color: #38bdf8;
            border-bottom: 3px solid #38bdf8;
            font-weight: 600;
        }}
        .nav-tabs .nav-link:hover {{
            border-bottom: 3px solid #1e293b;
            color: #f1f5f9;
        }}
        .quality-card {{
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
            border: 1px solid #312e81;
        }}
        .score-circle {{
            width: 80px;
            height: 80px;
            border-radius: 50%;
            border: 4px solid #38bdf8;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            background: rgba(56, 189, 248, 0.1);
        }}
    </style>
</head>
<body class="py-5">
    <div class="container pb-5">
        
        <!-- Header Panel -->
        <header class="d-flex justify-content-between align-items-center p-4 rounded-4 mb-4 glass shadow">
            <div>
                <h1 class="h3 mb-2 text-light fw-bold"><i class="bi bi-cpu-fill me-2 text-info"></i>Semiconductor Restorations Audit</h1>
                <p class="text-secondary mb-0 font-monospace" style="font-size:0.82rem;">
                    DATASET FINGERPRINT: <span class="text-info">{fingerprint_display}</span>
                </p>
            </div>
            <div class="d-flex align-items-center gap-3">
                <div class="text-end">
                    <span class="d-block mb-1 text-muted text-uppercase font-monospace small">Audit Status</span>
                    <span class="badge bg-{status_color} px-3 py-2 text-capitalize fs-6">{val_status}</span>
                </div>
                <div class="score-circle text-info fs-5" title="Dataset Quality Score (0-100)">
                    {val_score}%
                </div>
            </div>
        </header>

        <!-- Dynamic Content Tabs -->
        <ul class="nav nav-tabs mb-4 border-secondary" id="reportTabs" role="tablist">
            <li class="nav-item" role="presentation">
                <button class="nav-link active" id="overview-tab" data-bs-toggle="tab" data-bs-target="#overview" type="button" role="tab" aria-controls="overview" aria-selected="true"><i class="bi bi-grid-fill me-2"></i>Overview</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="validation-tab" data-bs-toggle="tab" data-bs-target="#validation" type="button" role="tab" aria-controls="validation" aria-selected="false"><i class="bi bi-shield-check me-2"></i>Validation Audit</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="stats-tab" data-bs-toggle="tab" data-bs-target="#stats" type="button" role="tab" aria-controls="stats" aria-selected="false"><i class="bi bi-graph-up me-2"></i>Grayscale Stats</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="profiler-tab" data-bs-toggle="tab" data-bs-target="#profiler" type="button" role="tab" aria-controls="profiler" aria-selected="false"><i class="bi bi-sliders me-2"></i>Degradation Profiler</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="previews-tab" data-bs-toggle="tab" data-bs-target="#previews" type="button" role="tab" aria-controls="previews" aria-selected="false"><i class="bi bi-images me-2"></i>Previews</button>
            </li>
        </ul>

        <div class="tab-content" id="reportTabsContent">
            
            <!-- OVERVIEW TAB -->
            <div class="tab-pane fade show active" id="overview" role="tabpanel" aria-labelledby="overview-tab">
                <div class="row">
                    <div class="col-lg-7">
                        <div class="card bg-dark-card border-secondary mb-4 shadow">
                            <div class="card-body">
                                <h4 class="card-title text-light mb-3">Dataset Registries</h4>
                                <div class="table-responsive">
                                    <table class="table table-dark table-hover mb-0">
                                        <thead>
                                            <tr>
                                                <th>Dataset</th>
                                                <th>Version</th>
                                                <th>Source</th>
                                                <th>SHA256 File Signature</th>
                                                <th class="text-end">Pairs Processed</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {datasets_list_html}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </div>

                        <!-- Metrology statistical Dashboard -->
                        <div class="card bg-dark-card border-secondary mb-4 shadow">
                            <div class="card-body">
                                <h4 class="card-title text-light mb-4">Core Dimensions Summary</h4>
                                <div class="row text-center font-monospace">
                                    <div class="col-md-4 mb-3">
                                        <div class="p-3 border rounded border-secondary bg-dark">
                                            <div class="text-secondary small">TOTAL MICRO-PAIRS</div>
                                            <div class="fs-3 fw-bold text-light">{stats.get("total_images", 0)//2}</div>
                                        </div>
                                    </div>
                                    <div class="col-md-4 mb-3">
                                        <div class="p-3 border rounded border-secondary bg-dark">
                                            <div class="text-secondary small">AVG RESOLUTION</div>
                                            <div class="fs-4 fw-bold text-light mt-1">
                                                {stats.get("dimensions", {}).get("avg_width", 0):.0f}×{stats.get("dimensions", {}).get("avg_height", 0):.0f}
                                            </div>
                                        </div>
                                    </div>
                                    <div class="col-md-4 mb-3">
                                        <div class="p-3 border rounded border-secondary bg-dark">
                                            <div class="text-secondary small">RESOLUTION PROFILE</div>
                                            <div class="fs-6 fw-bold text-light mt-2">
                                                Min: {stats.get("dimensions", {}).get("min_width", 0)}px<br>
                                                Max: {stats.get("dimensions", {}).get("max_width", 0)}px
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="col-lg-5">
                        <div class="card quality-card mb-4 shadow text-light">
                            <div class="card-body">
                                <h4 class="card-title mb-3">Dataset Quality Audit</h4>
                                <div class="d-flex align-items-center gap-3 mb-3">
                                    <div class="fs-1 fw-bold text-info">{val_score}/100</div>
                                    <div>
                                        <div class="fw-bold">Overall Score</div>
                                        <span class="badge bg-{status_color}">{val_status}</span>
                                    </div>
                                </div>
                                <p class="text-secondary small">
                                    The quality score evaluates dataset completeness, shape synchronization, duplicate files, name validation, and grayscaling. Any missing files or corruptions result in immediate failure.
                                </p>
                            </div>
                        </div>

                        <div class="card bg-dark-card border-secondary mb-4 shadow text-light">
                            <div class="card-body">
                                <h4 class="card-title mb-3">Split Distributions</h4>
                                <div class="list-group font-monospace small">
                                    """ + "\n".join([f"""
                                    <div class="list-group-item bg-dark border-secondary text-light d-flex justify-content-between align-items-center">
                                        <span class="text-info fw-bold">{sp.upper()}</span>
                                        <span>{sp_info.get("pairs_count", 0)} pairs</span>
                                    </div>
                                    """ for sp, sp_info in stats.get("splits_summary", {}).items()]) + """
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- VALIDATION TAB -->
            <div class="tab-pane fade" id="validation" role="tabpanel" aria-labelledby="validation-tab">
                <div class="row">
                    <div class="col-lg-6">
                        <div class="card bg-dark-card border-secondary mb-4 shadow">
                            <div class="card-body">
                                <h4 class="card-title text-light mb-3">Checklist Results</h4>
                                <ul class="list-group">
                                    {check_list_html}
                                </ul>
                            </div>
                        </div>
                    </div>
                    <div class="col-lg-6">
                        <div class="card bg-dark-card border-secondary mb-4 shadow">
                            <div class="card-body">
                                <h4 class="card-title text-light mb-3">Validation Sweep Log Details</h4>
                                <div class="p-3 border rounded border-secondary bg-dark font-monospace text-secondary" style="font-size:0.85rem; min-height: 250px; max-height: 400px; overflow-y:auto;">
                                    {errors_details_html}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- STATS TAB -->
            <div class="tab-pane fade" id="stats" role="tabpanel" aria-labelledby="stats-tab">
                <div class="row">
                    <div class="col-lg-7">
                        <div class="card bg-dark-card border-secondary mb-4 shadow h-100">
                            <div class="card-body d-flex flex-column">
                                <h4 class="card-title text-light mb-3">Intensity Distribution comparison</h4>
                                <div class="flex-grow-1 d-flex align-items-center justify-content-center" style="min-height: 350px;">
                                    <canvas id="histoChart" style="width:100%; height:100%;"></canvas>
                                </div>
                                <div class="text-secondary text-center small mt-3 font-monospace">
                                    Comparing intensity distributions: Noisy Input vs Clean Ground Truth
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="col-lg-5">
                        <div class="card bg-dark-card border-secondary mb-4 shadow">
                            <div class="card-body">
                                <h4 class="card-title text-light mb-3">Numerical Metrology</h4>
                                <div class="row text-center font-monospace">
                                    <div class="col-6 mb-3">
                                        <div class="p-3 border rounded border-secondary bg-dark">
                                            <div class="text-warning small mb-1">NOISY MEAN</div>
                                            <div class="fs-4 fw-bold text-warning">{stats.get("noisy_stats", {}).get("mean", 0):.4f}</div>
                                        </div>
                                    </div>
                                    <div class="col-6 mb-3">
                                        <div class="p-3 border rounded border-secondary bg-dark">
                                            <div class="text-warning small mb-1">NOISY STD</div>
                                            <div class="fs-4 fw-bold text-warning">{stats.get("noisy_stats", {}).get("std", 0):.4f}</div>
                                        </div>
                                    </div>
                                    <div class="col-6 mb-3">
                                        <div class="p-3 border rounded border-secondary bg-dark">
                                            <div class="text-success small mb-1">CLEAN MEAN</div>
                                            <div class="fs-4 fw-bold text-success">{stats.get("clean_stats", {}).get("mean", 0):.4f}</div>
                                        </div>
                                    </div>
                                    <div class="col-6 mb-3">
                                        <div class="p-3 border rounded border-secondary bg-dark">
                                            <div class="text-success small mb-1">CLEAN STD</div>
                                            <div class="fs-4 fw-bold text-success">{stats.get("clean_stats", {}).get("std", 0):.4f}</div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- PROFILER TAB -->
            <div class="tab-pane fade" id="profiler" role="tabpanel" aria-labelledby="profiler-tab">
                <div class="row">
                    <div class="col-lg-8">
                        <div class="card bg-dark-card border-secondary mb-4 shadow h-100">
                            <div class="card-body d-flex flex-column">
                                <h4 class="card-title text-light mb-3">Metrology Restoration Scores by Dataset</h4>
                                <div class="flex-grow-1 d-flex align-items-center justify-content-center" style="min-height: 350px;">
                                    <canvas id="profilerChart" style="width:100%; height:100%;"></canvas>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="col-lg-4">
                        <div class="card bg-dark-card border-secondary mb-4 shadow">
                            <div class="card-body">
                                <h4 class="card-title text-light mb-3">Global Metrics Summary</h4>
                                <ul class="list-group font-monospace small">
                                    <li class="list-group-item bg-dark border-secondary text-light d-flex justify-content-between py-3">
                                        <span class="text-secondary">Average PSNR:</span>
                                        <span class="text-info fw-bold">{profile_data.get("global_metrics", {}).get("avg_psnr", 0.0):.2f} dB</span>
                                    </li>
                                    <li class="list-group-item bg-dark border-secondary text-light d-flex justify-content-between py-3">
                                        <span class="text-secondary">Average SSIM:</span>
                                        <span class="text-success fw-bold">{profile_data.get("global_metrics", {}).get("avg_ssim", 0.0):.4f}</span>
                                    </li>
                                    <li class="list-group-item bg-dark border-secondary text-light d-flex justify-content-between py-3">
                                        <span class="text-secondary">Estimated Noise Std:</span>
                                        <span class="text-warning fw-bold">{profile_data.get("global_metrics", {}).get("avg_noise_level", 0.0):.4f}</span>
                                    </li>
                                    <li class="list-group-item bg-dark border-secondary text-light d-flex justify-content-between py-3">
                                        <span class="text-secondary">Avg Noisy Entropy:</span>
                                        <span class="text-light fw-bold">{profile_data.get("global_metrics", {}).get("avg_entropy_noisy", 0.0):.2f} bits</span>
                                    </li>
                                    <li class="list-group-item bg-dark border-secondary text-light d-flex justify-content-between py-3">
                                        <span class="text-secondary">Avg Edge Density:</span>
                                        <span class="text-light fw-bold">{profile_data.get("global_metrics", {}).get("avg_edge_density_noisy", 0.0):.4f}</span>
                                    </li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- PREVIEWS TAB -->
            <div class="tab-pane fade" id="previews" role="tabpanel" aria-labelledby="previews-tab">
                {sample_rows_html}
            </div>

        </div>

    </div>

    <!-- ChartJS and Bootstrap CDN bundles -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
        document.addEventListener("DOMContentLoaded", function() {{
            // 1. Dual Histograms Chart
            const ctxHist = document.getElementById('histoChart').getContext('2d');
            const data_bins_noisy = {noisy_hist_list};
            const data_bins_clean = {clean_hist_list};
            const labels = Array.from({{length: 256}}, (_, i) => i);
            
            new Chart(ctxHist, {{
                type: 'line',
                data: {{
                    labels: labels,
                    datasets: [
                        {{
                            label: 'Noisy Inputs',
                            data: data_bins_noisy,
                            borderColor: '#f97316',
                            backgroundColor: 'rgba(249, 115, 22, 0.08)',
                            fill: true,
                            pointRadius: 0,
                            borderWidth: 2,
                            tension: 0.4
                        }},
                        {{
                            label: 'Clean Targets',
                            data: data_bins_clean,
                            borderColor: '#38bdf8',
                            backgroundColor: 'rgba(56, 189, 248, 0.08)',
                            fill: true,
                            pointRadius: 0,
                            borderWidth: 2,
                            tension: 0.4
                        }}
                    ]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{ display: true, labels: {{ color: '#94a3b8' }} }},
                        tooltip: {{
                            backgroundColor: '#111b2d',
                            titleColor: '#f8fafc',
                            bodyColor: '#e2e8f0',
                            borderColor: '#1e293b',
                            borderWidth: 1,
                            callbacks: {{
                                title: function(context) {{
                                    return 'Intensity: ' + context[0].label;
                                }}
                            }}
                        }}
                    }},
                    scales: {{
                        x: {{
                            grid: {{ color: '#1e293b' }},
                            ticks: {{ color: '#64748b', font: {{ family: 'JetBrains Mono' }} }}
                        }},
                        y: {{
                            grid: {{ color: '#1e293b' }},
                            ticks: {{ color: '#64748b', font: {{ family: 'JetBrains Mono' }} }}
                        }}
                    }}
                }}
            }});

            // 2. Profiler Chart
            const ctxProfile = document.getElementById('profilerChart').getContext('2d');
            const ds_labels = {profile_labels};
            const ds_psnrs = {profile_psnr};
            const ds_ssims = {profile_ssim};
            
            new Chart(ctxProfile, {{
                type: 'bar',
                data: {{
                    labels: ds_labels,
                    datasets: [
                        {{
                            label: 'Average PSNR (dB)',
                            data: ds_psnrs,
                            backgroundColor: '#38bdf8',
                            borderColor: '#0284c7',
                            borderWidth: 1,
                            yAxisID: 'y'
                        }},
                        {{
                            label: 'Average SSIM',
                            data: ds_ssims,
                            backgroundColor: '#10b981',
                            borderColor: '#059669',
                            borderWidth: 1,
                            yAxisID: 'y1'
                        }}
                    ]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{ display: true, labels: {{ color: '#94a3b8' }} }}
                    }},
                    scales: {{
                        x: {{
                            grid: {{ color: '#1e293b' }},
                            ticks: {{ color: '#94a3b8', font: {{ family: 'JetBrains Mono' }} }}
                        }},
                        y: {{
                            type: 'linear',
                            position: 'left',
                            grid: {{ color: '#1e293b' }},
                            ticks: {{ color: '#38bdf8' }},
                            title: {{ display: true, text: 'PSNR (dB)', color: '#38bdf8' }}
                        }},
                        y1: {{
                            type: 'linear',
                            position: 'right',
                            grid: {{ drawOnChartArea: false }},
                            ticks: {{ color: '#10b981' }},
                            title: {{ display: true, text: 'SSIM', color: '#10b981' }},
                            min: 0,
                            max: 1.0
                        }}
                    }}
                }}
            }});
        }});
    </script>
</body>
</html>
"""
        # Save output HTML
        output_html_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        logger.info(f"Saved publication-quality HTML report to {output_html_path.absolute()}")
