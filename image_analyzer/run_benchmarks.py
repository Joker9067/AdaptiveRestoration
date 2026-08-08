"""
Benchmarking orchestrator for the Physics-Guided Image Analyzer (Module 7).
Trains all 3 backbone candidates (EfficientNet-B0, MobileNetV3-Small, ConvNeXt-Tiny),
profiles metrics (Acc, F1, MAE, R², FLOPs, Latency), ranks them, and outputs comparison reports.
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset_manager.config import PipelineConfig
from dataset_manager.manager import DatasetManager
from image_analyzer.config import AnalyzerConfig
from image_analyzer.analyzer_dataset import ImageAnalyzerDataset, NOISE_TYPE_MAP, BLUR_TYPE_MAP, SEVERITY_MAP
from image_analyzer.model import PhysicsImageAnalyzer
from image_analyzer.trainer import ImageAnalyzerTrainer

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s [%(name)s] - %(message)s")
logger = logging.getLogger(__name__)


# ----------------------------------------------------
# 1. Custom Scientific Metric Engines
# ----------------------------------------------------

def calculate_classification_metrics(preds: np.ndarray, targets: np.ndarray, num_classes: int) -> Tuple[float, float, float, float]:
    """Computes Accuracy, Macro Precision, Macro Recall, and Macro F1 score."""
    accuracy = float(np.mean(preds == targets))
    
    precisions = []
    recalls = []
    
    for c in range(num_classes):
        tp = np.sum((preds == c) & (targets == c))
        fp = np.sum((preds == c) & (targets != c))
        fn = np.sum((preds != c) & (targets == c))
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        precisions.append(prec)
        recalls.append(rec)
        
    macro_prec = float(np.mean(precisions))
    macro_rec = float(np.mean(recalls))
    
    if (macro_prec + macro_rec) > 0:
        macro_f1 = float(2.0 * (macro_prec * macro_rec) / (macro_prec + macro_rec))
    else:
        macro_f1 = 0.0
        
    return accuracy, macro_prec, macro_rec, macro_f1


def calculate_regression_metrics(preds: np.ndarray, targets: np.ndarray) -> Tuple[float, float, float]:
    """Computes MAE, RMSE, and R2 Score."""
    mae = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    
    # R2 Score
    target_mean = np.mean(targets)
    ss_tot = np.sum((targets - target_mean) ** 2)
    ss_res = np.sum((targets - preds) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 1.0
    
    return mae, rmse, r2


# ----------------------------------------------------
# 2. Dynamic HTML Dashboard Generation
# ----------------------------------------------------

def generate_html_report(results: List[Dict[str, Any]], reports_dir: Path) -> Path:
    out_path = reports_dir / "image_analyzer_dashboard.html"
    
    chart_labels = [r["model_name"].upper() for r in results]
    chart_accuracy = [r["avg_classification_accuracy"] for r in results]
    chart_f1 = [r["avg_classification_f1"] for r in results]
    chart_mae = [r["avg_regression_mae"] for r in results]
    chart_latency = [r["inference_time_ms"] for r in results]

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Image Analyzer Benchmarking Audit (Module 7)</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --bg-color: #0f172a;
            --card-bg: rgba(30, 41, 59, 0.7);
            --border-color: rgba(255, 255, 255, 0.1);
            --text-color: #f1f5f9;
            --accent-glow: #818cf8;
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
            background: linear-gradient(135deg, #818cf8, #a78bfa);
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
    </style>
</head>
<body>
    <div class="container">
        <h1>Image Analyzer Benchmarking Audit</h1>
        <p class="subtitle">Quantitative validation of multi-task degradation classification, regression, and footprint parameters</p>

        <div class="grid">
            <div class="card">
                <h2>Classification accuracy & F1 Score (%)</h2>
                <canvas id="accuracyChart"></canvas>
            </div>
            <div class="card">
                <h2>Regression MAE vs Inference Latency</h2>
                <canvas id="regressionChart"></canvas>
            </div>
        </div>

        <div class="card">
            <h2>Image Analyzer Backbone Comparison Table</h2>
            <table>
                <thead>
                    <tr>
                        <th>Model Name</th>
                        <th>Class Acc (%)</th>
                        <th>Class F1 (%)</th>
                        <th>Reg MAE</th>
                        <th>Reg R²</th>
                        <th>Latency (ms)</th>
                        <th>Parameters</th>
                        <th>FLOPs (M)</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(f'''<tr>
                        <td><strong>{r["model_name"].upper()}</strong></td>
                        <td>{r["avg_classification_accuracy"] * 100.0:.2f}%</td>
                        <td>{r["avg_classification_f1"] * 100.0:.2f}%</td>
                        <td>{r["avg_regression_mae"]:.4f}</td>
                        <td>{r["avg_regression_r2"]:.4f}</td>
                        <td>{r["inference_time_ms"]:.2f}ms</td>
                        <td>{r["parameter_count"]:,}</td>
                        <td>{r["flops"] / 1e6:.1f}M</td>
                    </tr>''' for r in results)}
                </tbody>
            </table>
        </div>
    </div>

    <script>
        // Accuracy Chart
        const aCtx = document.getElementById('accuracyChart').getContext('2d');
        new Chart(aCtx, {{
            type: 'bar',
            data: {{
                labels: {json.dumps(chart_labels)},
                datasets: [{{
                    label: 'Accuracy (%)',
                    data: {json.dumps([x * 100.0 for x in chart_accuracy])},
                    backgroundColor: '#818cf8',
                }}, {{
                    label: 'F1 Score (%)',
                    data: {json.dumps([x * 100.0 for x in chart_f1])},
                    backgroundColor: '#34d399',
                }}]
            }},
            options: {{
                responsive: true,
                scales: {{ y: {{ min: 0, max: 100 }} }}
            }}
        }});

        // Regression Chart
        const rCtx = document.getElementById('regressionChart').getContext('2d');
        new Chart(rCtx, {{
            type: 'bar',
            data: {{
                labels: {json.dumps(chart_labels)},
                datasets: [{{
                    label: 'MAE (lower is better)',
                    data: {json.dumps(chart_mae)},
                    backgroundColor: '#fb7185',
                    yAxisID: 'y'
                }}, {{
                    label: 'Latency (ms)',
                    data: {json.dumps(chart_latency)},
                    borderColor: '#f59e0b',
                    type: 'line',
                    yAxisID: 'y1',
                    fill: false
                }}]
            }},
            options: {{
                responsive: true,
                scales: {{
                    y: {{ type: 'linear', display: true, position: 'left' }},
                    y1: {{ type: 'linear', display: true, position: 'right', grid: {{ drawOnChartArea: false }} }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    logger.info(f"Saved HTML dashboard to: {out_path.absolute()}")
    return out_path


# ----------------------------------------------------
# 3. Backbone Evaluation Process
# ----------------------------------------------------

def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """Runs evaluation and extracts classification metrics and regression metrics."""
    model.eval()
    
    # Classification Accumulators
    cls_keys = ["noise_type", "blur_type", "severity"]
    cls_num_classes = {
        "noise_type": len(NOISE_TYPE_MAP),
        "blur_type": len(BLUR_TYPE_MAP),
        "severity": len(SEVERITY_MAP)
    }
    
    cls_preds: Dict[str, List[int]] = {k: [] for k in cls_keys}
    cls_tgts: Dict[str, List[int]] = {k: [] for k in cls_keys}

    # Regression Accumulators
    reg_keys = [
        "noise_level", "blur_strength", "resolution_loss", "compression_quality",
        "brightness", "contrast", "gamma", "edge_density", "texture_complexity", "entropy", "confidence"
    ]
    reg_preds: Dict[str, List[float]] = {k: [] for k in reg_keys}
    reg_tgts: Dict[str, List[float]] = {k: [] for k in reg_keys}

    # Latency timing
    latency_records = []

    with torch.no_grad():
        for inputs, _, targets in loader:
            inputs = inputs.to(device)
            batch_size = inputs.size(0)

            # Profile latency
            t_start = time.perf_counter()
            outputs = model(inputs)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t_duration = time.perf_counter() - t_start
            latency_records.append(t_duration / batch_size)

            # Classification Predictions
            for k in cls_keys:
                logits = outputs[k]
                preds = torch.argmax(logits, dim=1)
                cls_preds[k].extend(preds.cpu().numpy().tolist())
                cls_tgts[k].extend(targets[k].numpy().tolist())

            # Regression Predictions
            for k in reg_keys:
                reg_preds[k].extend(outputs[k].cpu().numpy().tolist())
                reg_tgts[k].extend(targets[k].numpy().tolist())

    # Formulate Classification Metrics
    cls_metrics = {}
    for k in cls_keys:
        acc, prec, rec, f1 = calculate_classification_metrics(
            np.array(cls_preds[k]),
            np.array(cls_tgts[k]),
            cls_num_classes[k]
        )
        cls_metrics[f"{k}_accuracy"] = acc
        cls_metrics[f"{k}_f1"] = f1

    # Formulate Regression Metrics
    reg_metrics = {}
    for k in reg_keys:
        mae, rmse, r2 = calculate_regression_metrics(
            np.array(reg_preds[k]),
            np.array(reg_tgts[k])
        )
        reg_metrics[f"{k}_mae"] = mae
        reg_metrics[f"{k}_rmse"] = rmse
        reg_metrics[f"{k}_r2"] = r2

    # Averages
    avg_cls_acc = np.mean([cls_metrics[f"{k}_accuracy"] for k in cls_keys])
    avg_cls_f1 = np.mean([cls_metrics[f"{k}_f1"] for k in cls_keys])
    avg_reg_mae = np.mean([reg_metrics[f"{k}_mae"] for k in reg_keys])
    avg_reg_rmse = np.mean([reg_metrics[f"{k}_rmse"] for k in reg_keys])
    avg_reg_r2 = np.mean([reg_metrics[f"{k}_r2"] for k in reg_keys])
    avg_latency_ms = np.mean(latency_records) * 1000.0

    metrics = {
        "avg_classification_accuracy": float(avg_cls_acc),
        "avg_classification_f1": float(avg_cls_f1),
        "avg_regression_mae": float(avg_reg_mae),
        "avg_regression_rmse": float(avg_reg_rmse),
        "avg_regression_r2": float(avg_reg_r2),
        "inference_time_ms": float(avg_latency_ms)
    }

    # Add detail keys
    metrics.update(cls_metrics)
    metrics.update(reg_metrics)
    return metrics


# ----------------------------------------------------
# 4. Orchestration Loop
# ----------------------------------------------------

def run_benchmarks() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Initializing Physics-Guided Image Analyzer Benchmarking...")

    workspace_path = Path(".").resolve()
    pipeline_cfg_path = workspace_path / "config.yaml"
    pipeline_cfg = PipelineConfig.load_from_yaml(pipeline_cfg_path)
    
    analyzer_cfg = AnalyzerConfig.load_from_yaml(pipeline_cfg_path)
    reports_dir = Path(analyzer_cfg.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Loading Datasets
    manager = DatasetManager(pipeline_cfg)
    
    raw_train_set = manager.load(dataset_name="Synthetic_Generated", split="train", is_training=True)
    raw_val_set = manager.load(dataset_name="Synthetic_Generated", split="val", is_training=False)

    # Wrap in Multi-Task analyzer dataset (resized to 128x128 for CPU training)
    train_set = ImageAnalyzerDataset(raw_train_set.df, manager.metadata_csv_path.parent.parent.parent, image_size=(128, 128))
    val_set = ImageAnalyzerDataset(raw_val_set.df, manager.metadata_csv_path.parent.parent.parent, image_size=(128, 128))

    train_loader = DataLoader(train_set, batch_size=analyzer_cfg.hyperparameters.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=analyzer_cfg.hyperparameters.batch_size, shuffle=False)

    backbones = ["efficientnet_b0", "mobilenet_v3", "convnext_tiny"]
    results = []

    for name in backbones:
        logger.info(f"\nTraining and benchmarking backbone: '{name}'...")
        
        # Instantiate model
        model = PhysicsImageAnalyzer(backbone_name=name)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=analyzer_cfg.hyperparameters.learning_rate)
        
        trainer = ImageAnalyzerTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=None,
            config=analyzer_cfg,
            device=device,
            model_name=name
        )
        trainer.train()

        # Load weights from best checkpoint
        best_pth = Path(analyzer_cfg.checkpoint_dir) / name / "best.pt"
        checkpoint = torch.load(best_pth, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

        # Evaluate
        logger.info(f"Evaluating best checkpoint for '{name}'...")
        eval_metrics = evaluate_model(model, val_loader, device)

        # Footprint complexities
        param_count = model.get_parameter_count()
        flops = model.get_flops(input_size=(1, 1, 128, 128))

        record = {
            "model_name": name,
            "parameter_count": param_count,
            "flops": flops,
            **eval_metrics
        }
        results.append(record)

        logger.info(
            f"Backbone '{name}' finished. "
            f"Acc: {eval_metrics['avg_classification_accuracy'] * 100.0:.2f}% | "
            f"F1: {eval_metrics['avg_classification_f1'] * 100.0:.2f}% | "
            f"MAE: {eval_metrics['avg_regression_mae']:.4f} | "
            f"Latency: {eval_metrics['inference_time_ms']:.2f}ms"
        )

    # 2. Rankings & Auto-Selection of the best backbone
    # Ranked by: Avg Class Accuracy + Avg Class F1 - Regression MAE - Latency/200
    def rank_key(x):
        return x["avg_classification_accuracy"] + x["avg_classification_f1"] - x["avg_regression_mae"] - (x["inference_time_ms"] / 200.0)

    ranked_results = sorted(results, key=rank_key, reverse=True)
    best_model_name = ranked_results[0]["model_name"]
    logger.info(f"\n--> Best performing backbone identified: '{best_model_name}'")

    # Save output files
    # A. JSON results
    json_path = reports_dir / "image_analyzer_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ranked_results, f, indent=4)
    logger.info(f"Saved JSON results to: {json_path.absolute()}")

    # B. CSV results
    df = pd.DataFrame(ranked_results)
    csv_path = reports_dir / "image_analyzer_results.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV results to: {csv_path.absolute()}")

    # C. Dynamic HTML Dashboard
    generate_html_report(ranked_results, reports_dir)

    # Copy the best model checkpoint to best_analyzer.pt in checkpoint root
    best_weights_src = Path(analyzer_cfg.checkpoint_dir) / best_model_name / "best.pt"
    best_weights_dst = Path(analyzer_cfg.checkpoint_dir) / "best_analyzer.pt"
    
    import shutil
    shutil.copy(best_weights_src, best_weights_dst)
    logger.info(f"Copied best model weights to: {best_weights_dst.absolute()}")


if __name__ == "__main__":
    run_benchmarks()
