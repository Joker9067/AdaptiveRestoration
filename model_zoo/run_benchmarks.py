"""
Model Zoo Benchmarking orchestrator for the Semiconductor Image Restoration System (Module 6).
Trains/loads all 8 restoration models, profiles latency and metrics, ranks performance,
and outputs CSV, JSON, and HTML dashboard reports.
"""

import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset_manager.config import PipelineConfig
from dataset_manager.manager import DatasetManager
from model_zoo.common.losses import CombinedWeightedLoss
from model_zoo.common.trainer import ModelTrainer
from model_zoo.common.evaluator import ModelEvaluator

# Model import maps
from model_zoo.dncnn.model import DnCNN
from model_zoo.ridnet.model import RIDNet
from model_zoo.nafnet.model import NAFNet
from model_zoo.edsr.model import EDSR
from model_zoo.rcan.model import RCAN
from model_zoo.swinir.model import SwinIR
from model_zoo.restormer.model import Restormer
from model_zoo.unet.model import UNet

# Config loaders
import model_zoo.dncnn.config as dncnn_cfg
import model_zoo.ridnet.config as ridnet_cfg
import model_zoo.nafnet.config as nafnet_cfg
import model_zoo.edsr.config as edsr_cfg
import model_zoo.rcan.config as rcan_cfg
import model_zoo.swinir.config as swinir_cfg
import model_zoo.restormer.config as restormer_cfg
import model_zoo.unet.config as unet_cfg

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s [%(name)s] - %(message)s")
logger = logging.getLogger(__name__)

# Category registry
CATEGORIES = {
    "dncnn": "Noise Restoration",
    "ridnet": "Noise Restoration",
    "nafnet": "Noise Restoration",
    "edsr": "Super Resolution",
    "rcan": "Super Resolution",
    "swinir": "Super Resolution",
    "restormer": "Structure Restoration",
    "unet": "Structure Restoration"
}

MODEL_CLASSES = {
    "dncnn": DnCNN,
    "ridnet": RIDNet,
    "nafnet": NAFNet,
    "edsr": EDSR,
    "rcan": RCAN,
    "swinir": SwinIR,
    "restormer": Restormer,
    "unet": UNet
}

CONFIG_LOADERS = {
    "dncnn": dncnn_cfg.get_config,
    "ridnet": ridnet_cfg.get_config,
    "nafnet": nafnet_cfg.get_config,
    "edsr": edsr_cfg.get_config,
    "rcan": rcan_cfg.get_config,
    "swinir": swinir_cfg.get_config,
    "restormer": restormer_cfg.get_config,
    "unet": unet_cfg.get_config
}


def build_ranking_scores(results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Sorts and ranks models inside each functional category (Noise, SR, Structure)."""
    rankings_by_category = {}
    
    # Group by category
    for cat_name in set(CATEGORIES.values()):
        cat_results = [r for r in results if r["category"] == cat_name]
        
        # Rank by combined metric score: PSNR (higher is better) + 10 * SSIM (higher is better) - (Latency_ms / 20)
        # Weighting performance and speed
        def rank_key(x):
            return x["psnr"] + 10.0 * x["ssim"] - (x["inference_time_ms"] / 100.0)

        sorted_results = sorted(cat_results, key=rank_key, reverse=True)
        
        ranked_list = []
        for rank, model_data in enumerate(sorted_results, 1):
            ranked_list.append({
                "rank": rank,
                "model_name": model_data["model_name"],
                "psnr": model_data["psnr"],
                "ssim": model_data["ssim"],
                "lpips": model_data["lpips"],
                "inference_time_ms": model_data["inference_time_ms"],
                "parameter_count": model_data["parameter_count"],
                "flops": model_data["flops"]
            })
        rankings_by_category[cat_name] = ranked_list

    return rankings_by_category


def generate_html_report(results: List[Dict[str, Any]], rankings: Dict[str, List[Dict[str, Any]]], reports_dir: Path) -> Path:
    """Generates a premium, publication-quality dark-theme HTML report with Chart.js visualization."""
    out_path = reports_dir / "benchmark_report.html"
    
    # Dynamic serialization for frontend Chart.js ingestion
    chart_labels = [r["model_name"].upper() for r in results]
    chart_psnr = [r["psnr"] for r in results]
    chart_ssim = [r["ssim"] for r in results]
    chart_latency = [r["inference_time_ms"] for r in results]
    chart_params = [r["parameter_count"] / 1e6 for r in results] # Millions

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Model Zoo Benchmark & Performance Audit</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --bg-color: #0f172a;
            --card-bg: rgba(30, 41, 59, 0.7);
            --border-color: rgba(255, 255, 255, 0.1);
            --text-color: #f1f5f9;
            --accent-glow: #38bdf8;
            --accent-success: #34d399;
            --accent-warning: #fb7185;
        }}
        body {{
            background-color: var(--bg-color);
            color: var(--text-color);
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            margin: 0;
            padding: 40px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{
            font-size: 2.5rem;
            font-weight: 800;
            text-align: center;
            background: linear-gradient(135deg, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 5px;
        }}
        .subtitle {{
            text-align: center;
            color: #94a3b8;
            margin-bottom: 40px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 40px;
        }}
        .card {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            backdrop-filter: blur(16px);
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.4);
        }}
        .card h2 {{
            margin-top: 0;
            font-size: 1.25rem;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 10px;
            color: var(--accent-glow);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}
        th {{
            color: #94a3b8;
            font-weight: 600;
        }}
        .rank-badge {{
            display: inline-block;
            width: 24px;
            height: 24px;
            border-radius: 50%;
            text-align: center;
            line-height: 24px;
            font-weight: bold;
        }}
        .rank-1 {{ background-color: #f59e0b; color: #000; }}
        .rank-2 {{ background-color: #cbd5e1; color: #000; }}
        .rank-3 {{ background-color: #b45309; color: #fff; }}
        .rank-other {{ background-color: var(--border-color); color: var(--text-color); }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Model Zoo Benchmark & Performance Audit</h1>
        <p class="subtitle">Quantitative validation of restoration quality, parameters efficiency, and execution latency across splits</p>

        <div class="grid">
            <div class="card">
                <h2>Restoration Quality (PSNR vs Latency)</h2>
                <canvas id="qualityChart"></canvas>
            </div>
            <div class="card">
                <h2>Model Complexity (Parameters in Millions)</h2>
                <canvas id="complexityChart"></canvas>
            </div>
        </div>

        <div class="card" style="margin-bottom: 30px;">
            <h2>Unified Benchmark Performance Table</h2>
            <table>
                <thead>
                    <tr>
                        <th>Model Name</th>
                        <th>Category</th>
                        <th>PSNR (dB)</th>
                        <th>SSIM</th>
                        <th>LPIPS</th>
                        <th>Inference (ms)</th>
                        <th>GPU Mem (MB)</th>
                        <th>Params</th>
                        <th>FLOPs (M)</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(f'''<tr>
                        <td><strong>{r["model_name"].upper()}</strong></td>
                        <td>{r["category"]}</td>
                        <td>{r["psnr"]:.2f}</td>
                        <td>{r["ssim"]:.4f}</td>
                        <td>{r["lpips"]:.4f}</td>
                        <td>{r["inference_time_ms"]:.2f}</td>
                        <td>{r["gpu_memory_mb"]:.1f}</td>
                        <td>{r["parameter_count"]:,}</td>
                        <td>{r["flops"] / 1e6:.1f}</td>
                    </tr>''' for r in results)}
                </tbody>
            </table>
        </div>

        {"".join(f'''<div class="card" style="margin-bottom: 20px;">
            <h2>Category Rankings: {cat}</h2>
            <table>
                <thead>
                    <tr>
                        <th style="width: 80px;">Rank</th>
                        <th>Model Name</th>
                        <th>PSNR (dB)</th>
                        <th>SSIM</th>
                        <th>LPIPS</th>
                        <th>Latency (ms)</th>
                        <th>Params</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(f'''<tr>
                        <td><span class="rank-badge rank-{"1" if item["rank"]==1 else ("2" if item["rank"]==2 else ("3" if item["rank"]==3 else "other"))}">{item["rank"]}</span></td>
                        <td><strong>{item["model_name"].upper()}</strong></td>
                        <td>{item["psnr"]:.2f}</td>
                        <td>{item["ssim"]:.4f}</td>
                        <td>{item["lpips"]:.4f}</td>
                        <td>{item["inference_time_ms"]:.2f}</td>
                        <td>{item["parameter_count"]:,}</td>
                    </tr>''' for item in items)}
                </tbody>
            </table>
        </div>''' for cat, items in rankings.items())}

    </div>

    <script>
        // Quality Chart
        const qCtx = document.getElementById('qualityChart').getContext('2d');
        new Chart(qCtx, {{
            type: 'bar',
            data: {{
                labels: {json.dumps(chart_labels)},
                datasets: [{{
                    label: 'PSNR (dB)',
                    data: {json.dumps(chart_psnr)},
                    backgroundColor: '#38bdf8',
                    yAxisID: 'y'
                }}, {{
                    label: 'Latency (ms)',
                    data: {json.dumps(chart_latency)},
                    borderColor: '#fb7185',
                    type: 'line',
                    yAxisID: 'y1',
                    fill: false
                }}]
            }},
            options: {{
                responsive: true,
                scales: {{
                    y: {{
                        type: 'linear',
                        display: true,
                        position: 'left',
                        grid: {{ color: 'rgba(255,255,255,0.05)' }}
                    }},
                    y1: {{
                        type: 'linear',
                        display: true,
                        position: 'right',
                        grid: {{ drawOnChartArea: false }}
                    }}
                }}
            }}
        }});

        // Complexity Chart
        const cCtx = document.getElementById('complexityChart').getContext('2d');
        new Chart(cCtx, {{
            type: 'doughnut',
            data: {{
                labels: {json.dumps(chart_labels)},
                datasets: [{{
                    label: 'Params (Millions)',
                    data: {json.dumps(chart_params)},
                    backgroundColor: [
                        '#38bdf8', '#818cf8', '#a78bfa', '#fb7185',
                        '#fb923c', '#fbbf24', '#34d399', '#2dd4bf'
                    ]
                }}]
            }},
            options: {{
                responsive: true
            }}
        }});
    </script>
</body>
</html>
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    logger.info(f"Saved HTML benchmark report to: {out_path.absolute()}")
    return out_path


class ResizeDataset(torch.utils.data.Dataset):
    """Wrapper to resize dataset images to a lower size for fast CPU dry-runs."""
    def __init__(self, dataset, size=(128, 128)):
        self.dataset = dataset
        self.size = size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        x_resized = torch.nn.functional.interpolate(x.unsqueeze(0), size=self.size, mode="bilinear", align_corners=False).squeeze(0)
        y_resized = torch.nn.functional.interpolate(y.unsqueeze(0), size=self.size, mode="bilinear", align_corners=False).squeeze(0)
        return x_resized, y_resized


def run_benchmark() -> None:
    # 1. Resolve configurations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting benchmarking orchestrator on {device}...")

    workspace_path = Path(".").resolve()
    pipeline_cfg_path = workspace_path / "config.yaml"
    if not pipeline_cfg_path.exists():
        raise FileNotFoundError(f"Configuration file config.yaml not found.")
    pipeline_cfg = PipelineConfig.load_from_yaml(pipeline_cfg_path)

    reports_dir = Path(pipeline_cfg.paths.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 2. Ingest datasets using DatasetManager
    manager = DatasetManager(pipeline_cfg)
    if not (Path(pipeline_cfg.paths.processed_dir) / "metadata.csv").exists():
        logger.info("No processed dataset found. Executing preprocessing pipeline step...")
        manager.prepare_and_process_all()

    # Load validation split of Synthetic_Generated dataset and resize for speed
    raw_val_set = manager.load(dataset_name="Synthetic_Generated", split="val", is_training=False)
    val_set = ResizeDataset(raw_val_set, size=(128, 128))
    val_loader = DataLoader(val_set, batch_size=2, shuffle=False, num_workers=0, pin_memory=False)

    results: List[Dict[str, Any]] = []

    # 3. Benchmark loop
    for model_name, model_cls in MODEL_CLASSES.items():
        logger.info(f"\nEvaluating model candidate: '{model_name}'...")
        cat = CATEGORIES[model_name]

        # Resolve config dict
        cfg_loader = CONFIG_LOADERS[model_name]
        cfg = cfg_loader()

        # Instantiate model architecture
        model = model_cls()

        # Force a quick dry-run training epoch to generate best.pt checkpoint if missing
        checkpoint_dir = Path(cfg.get("checkpoint_dir", "./checkpoints")) / model_name
        best_pth = checkpoint_dir / "best.pt"
        
        if not best_pth.exists():
            logger.info(f"Checkpoint for '{model_name}' missing. Initiating dry-run training epoch to generate weights...")
            # Short training epoch overrides
            cfg["epochs"] = 1
            cfg["early_stopping_patience"] = 1
            
            # Sub-sample train loader for speed
            train_set = manager.load(dataset_name="Synthetic_Generated", split="train", is_training=True)
            # Take a small subset (e.g. 4 samples) and resize to run extremely fast
            train_subset_set = ResizeDataset(
                torch.utils.data.Subset(train_set, list(range(min(4, len(train_set))))),
                size=(128, 128)
            )
            train_loader = DataLoader(train_subset_set, batch_size=2, shuffle=True, num_workers=0)
            
            opt_name = cfg.get("optimizer", "Adam").lower()
            if opt_name == "adam":
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            else:
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

            loss_fn = CombinedWeightedLoss(cfg.get("loss_weights", {"l1": 1.0}))
            
            ds_cfg = pipeline_cfg.datasets.get("synthetic_generated")
            ds_version = ds_cfg.version if ds_cfg else "1.0"

            trainer = ModelTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                scheduler=None,
                loss_fn=loss_fn,
                config=cfg,
                device=device,
                model_name=model_name,
                dataset_version=ds_version,
                seed=42
            )
            trainer.train()

        # 4. Load weights from best checkpoint
        checkpoint = torch.load(best_pth, map_location=device)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        
        # 5. Run evaluation profiling
        evaluator = ModelEvaluator(model, val_loader, device)
        eval_metrics = evaluator.evaluate(input_size=(1, 1, 128, 128))
        
        # Formulate benchmark entry record
        record = {
            "model_name": model_name,
            "category": cat,
            "psnr": eval_metrics["psnr"],
            "ssim": eval_metrics["ssim"],
            "lpips": eval_metrics["lpips"],
            "inference_time_ms": eval_metrics["inference_time_ms"],
            "gpu_memory_mb": eval_metrics["gpu_memory_mb"],
            "parameter_count": eval_metrics["parameter_count"],
            "flops": eval_metrics["flops"]
        }
        results.append(record)
        
        logger.info(
            f"Candidate '{model_name}' - PSNR: {record['psnr']:.2f}dB | SSIM: {record['ssim']:.4f} | "
            f"Latency: {record['inference_time_ms']:.2f}ms | Params: {record['parameter_count']:,} | "
            f"FLOPs: {record['flops'] / 1e6:.2f}M"
        )

    # 6. Save outputs
    # A. CSV Output
    df = pd.DataFrame(results)
    csv_path = reports_dir / "benchmark_results.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved benchmark CSV results to: {csv_path.absolute()}")

    # B. JSON Output
    json_path = reports_dir / "benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Saved benchmark JSON results to: {json_path.absolute()}")

    # C. Sorted rankings JSON
    rankings = build_ranking_scores(results)
    rankings_path = reports_dir / "model_rankings.json"
    with open(rankings_path, "w", encoding="utf-8") as f:
        json.dump(rankings, f, indent=4)
    logger.info(f"Saved model rankings details to: {rankings_path.absolute()}")

    # D. HTML Dashboard Report
    generate_html_report(results, rankings, reports_dir)


if __name__ == "__main__":
    run_benchmark()
