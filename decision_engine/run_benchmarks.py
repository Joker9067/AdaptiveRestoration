"""
Benchmarking and optimization runner for the Adaptive Decision Engine (Module 8).
Trains the gating networks, resolves dynamic sequences, profiles comparative PSNR/SSIM gains,
saves routing reports, and outputs decision heatmaps and HTML dashboard.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from dataset_manager.config import PipelineConfig
from dataset_manager.manager import DatasetManager
from image_analyzer.analyzer_dataset import ImageAnalyzerDataset, NOISE_TYPE_MAP, BLUR_TYPE_MAP, SEVERITY_MAP
from image_analyzer.model import PhysicsImageAnalyzer
from model_zoo.dncnn.model import DnCNN
from model_zoo.rcan.model import RCAN
from model_zoo.unet.model import UNet
from decision_engine.config import DecisionEngineConfig
from decision_engine.model import RestorationPipeline, ORDER_PERMUTATIONS
from decision_engine.trainer import DecisionEngineTrainer
from model_zoo.common.metrics import calculate_psnr, calculate_ssim

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s [%(name)s] - %(message)s")
logger = logging.getLogger(__name__)


# Helper decoder maps
REV_NOISE_MAP = {v: k for k, v in NOISE_TYPE_MAP.items()}
REV_BLUR_MAP = {v: k for k, v in BLUR_TYPE_MAP.items()}
REV_SEVERITY_MAP = {v: k for k, v in SEVERITY_MAP.items()}


# ----------------------------------------------------
# 1. Visualization Generators
# ----------------------------------------------------

def generate_decision_heatmap(reports_dir: Path, weights: np.ndarray, noise_levels: np.ndarray, blur_levels: np.ndarray) -> Path:
    """Saves a 2D scatter plot heatmap showing expert weights as a function of noise & blur levels."""
    plt.figure(figsize=(8, 6))
    
    # Classify which expert dominates for each point
    dominant_expert = np.argmax(weights, axis=1)
    
    colors = ['#fb7185', '#38bdf8', '#34d399'] # Pink (Noise), Blue (SR), Green (Struct)
    labels = ['Noise Expert', 'SR Expert', 'Structure Expert']
    
    for i in range(3):
        mask = dominant_expert == i
        if np.any(mask):
            # Scale marker size based on the strength of the weight
            sizes = weights[mask, i] * 200.0 + 50.0
            plt.scatter(
                noise_levels[mask],
                blur_levels[mask],
                c=colors[i],
                label=labels[i],
                s=sizes,
                alpha=0.8,
                edgecolors='white',
                linewidths=1.0
            )

    plt.title("Expert Allocation Map (Dominant Experts vs Degradations)", fontsize=12, fontweight='bold', pad=15)
    plt.xlabel("Estimated Noise Level", fontsize=10)
    plt.ylabel("Estimated Blur Strength", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(frameon=True, facecolor='#1e293b', edgecolor='none', labelcolor='white')
    
    # Custom dark styling parameters
    plt.gca().set_facecolor('#0f172a')
    plt.gcf().patch.set_facecolor('#0f172a')
    plt.gca().tick_params(colors='white')
    plt.gca().xaxis.label.set_color('white')
    plt.gca().yaxis.label.set_color('white')
    plt.gca().title.set_color('white')
    for spine in plt.gca().spines.values():
        spine.set_color((1.0, 1.0, 1.0, 0.1))

    out_path = reports_dir / "expert_weight_heatmap.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0f172a')
    plt.close()
    logger.info(f"Saved expert allocation heatmap to: {out_path.absolute()}")
    return out_path


# ----------------------------------------------------
# 2. Dynamic HTML Gating Dashboard Generation
# ----------------------------------------------------

def generate_html_dashboard(reports_dir: Path, results: List[Dict[str, Any]]) -> Path:
    out_path = reports_dir / "decision_engine_dashboard.html"
    
    # Prepare serializations
    table_rows = []
    chart_labels = []
    chart_w_noise = []
    chart_w_sr = []
    chart_w_struct = []
    
    for r in results:
        img_name = r["image_name"]
        w = r["expert_weights"]
        seq = " &rarr; ".join(r["processing_order"])
        
        chart_labels.append(img_name)
        chart_w_noise.append(w["noise"])
        chart_w_sr.append(w["sr"])
        chart_w_struct.append(w["struct"])

        table_rows.append(f"""<tr>
            <td><strong>{img_name}</strong></td>
            <td>{r["noise_type"]} ({r["noise_level"]:.2f})</td>
            <td>{r["blur_type"]} ({r["blur_strength"]:.2f})</td>
            <td>{r["severity"]}</td>
            <td>
                <span class="weight-tag" style="background:#fb7185;color:#000;">N: {w["noise"]:.2f}</span>
                <span class="weight-tag" style="background:#38bdf8;color:#000;">SR: {w["sr"]:.2f}</span>
                <span class="weight-tag" style="background:#34d399;color:#000;">S: {w["struct"]:.2f}</span>
            </td>
            <td><code class="order-code">{seq}</code></td>
            <td>{r["confidence"]:.2f}</td>
            <td><strong>{r["fused_psnr"]:.2f} dB</strong> (vs {r["expert_psnr_baseline"]:.2f} dB best expert)</td>
        </tr>""")

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Adaptive Decision Engine Optimization Dashboard (Module 8)</title>
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
            background: linear-gradient(135deg, #34d399, #38bdf8);
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
            grid-template-columns: 2fr 1fr;
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
            color: #38bdf8;
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
        .weight-tag {{
            display: inline-block;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-right: 4px;
        }}
        .order-code {{
            background-color: rgba(0,0,0,0.3);
            padding: 4px 8px;
            border-radius: 6px;
            font-size: 0.85rem;
        }}
        .heatmap-img {{
            width: 100%;
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Adaptive Decision Engine Router Dashboard</h1>
        <p class="subtitle">Real-time explainable expert weighting and dynamic execution routing maps (Module 8)</p>

        <div class="grid">
            <div class="card">
                <h2>Dynamic Gating Weights Routing Breakdown</h2>
                <canvas id="gatingChart"></canvas>
            </div>
            <div class="card">
                <h2>Expert Allocation Map</h2>
                <img src="expert_weight_heatmap.png" class="heatmap-img" alt="Expert allocation heatmap">
            </div>
        </div>

        <div class="card" style="margin-top: 20px;">
            <h2>Explainable Routing Table</h2>
            <table>
                <thead>
                    <tr>
                        <th>Image Name</th>
                        <th>Noise State</th>
                        <th>Blur State</th>
                        <th>Severity</th>
                        <th>Expert Weights</th>
                        <th>Execution Sequence</th>
                        <th>Conf</th>
                        <th>Fused Restoration Output</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(table_rows)}
                </tbody>
            </table>
        </div>
    </div>

    <script>
        const gCtx = document.getElementById('gatingChart').getContext('2d');
        new Chart(gCtx, {{
            type: 'bar',
            data: {{
                labels: {json.dumps(chart_labels)},
                datasets: [{{
                    label: 'Noise Expert',
                    data: {json.dumps(chart_w_noise)},
                    backgroundColor: '#fb7185',
                }}, {{
                    label: 'SR Expert',
                    data: {json.dumps(chart_w_sr)},
                    backgroundColor: '#38bdf8',
                }}, {{
                    label: 'Structure Expert',
                    data: {json.dumps(chart_w_struct)},
                    backgroundColor: '#34d399',
                }}]
            }},
            options: {{
                responsive: true,
                scales: {{
                    x: {{ stacked: true, grid: {{ color: 'rgba(255,255,255,0.05)' }} }},
                    y: {{ stacked: true, min: 0, max: 1.0, grid: {{ color: 'rgba(255,255,255,0.05)' }} }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    logger.info(f"Saved Decision Engine HTML dashboard to: {out_path.absolute()}")
    return out_path


# ----------------------------------------------------
# 3. Main Orchestration Run
# ----------------------------------------------------

def run_benchmarks() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Initializing Decision Engine optimization process...")

    workspace_path = Path(".").resolve()
    pipeline_cfg_path = workspace_path / "config.yaml"
    pipeline_cfg = PipelineConfig.load_from_yaml(pipeline_cfg_path)
    
    de_cfg = DecisionEngineConfig.load_from_yaml(pipeline_cfg_path)
    reports_dir = Path(de_cfg.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ingest datasets
    manager = DatasetManager(pipeline_cfg)
    raw_train_set = manager.load(dataset_name="Synthetic_Generated", split="train", is_training=True)
    raw_val_set = manager.load(dataset_name="Synthetic_Generated", split="val", is_training=False)

    train_set = ImageAnalyzerDataset(raw_train_set.df, manager.metadata_csv_path.parent.parent.parent, image_size=(128, 128))
    val_set = ImageAnalyzerDataset(raw_val_set.df, manager.metadata_csv_path.parent.parent.parent, image_size=(128, 128))

    train_loader = DataLoader(train_set, batch_size=de_cfg.hyperparameters.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=de_cfg.hyperparameters.batch_size, shuffle=False)

    # 2. Instantiate and load pre-trained components
    # A. Image Analyzer
    analyzer = PhysicsImageAnalyzer(backbone_name="efficientnet_b0")
    analyzer_weights_pth = Path("checkpoints/analyzer/best_analyzer.pt")
    checkpoint = torch.load(analyzer_weights_pth, map_location=device, weights_only=False)
    analyzer.load_state_dict(checkpoint["model_state_dict"])
    analyzer = analyzer.eval()

    # B. Expert Zoo Models
    dncnn = DnCNN().eval()
    checkpoint_dncnn = torch.load("checkpoints/dncnn/best.pt", map_location=device, weights_only=False)
    dncnn.load_state_dict(checkpoint_dncnn.get("model_state_dict", checkpoint_dncnn))

    rcan = RCAN(upscale_factor=2).eval()
    checkpoint_rcan = torch.load("checkpoints/rcan/best.pt", map_location=device, weights_only=False)
    rcan.load_state_dict(checkpoint_rcan.get("model_state_dict", checkpoint_rcan))

    unet = UNet().eval()
    checkpoint_unet = torch.load("checkpoints/unet/best.pt", map_location=device, weights_only=False)
    unet.load_state_dict(checkpoint_unet.get("model_state_dict", checkpoint_unet))

    # C. Instantiate the differentiable sequential pipeline
    pipeline = RestorationPipeline(
        expert_noise=dncnn,
        expert_sr=rcan,
        expert_struct=unet,
        analyzer=analyzer
    )

    # 3. Train Decision Engine
    optimizer = torch.optim.AdamW(
        list(pipeline.gating_net.parameters()) + list(pipeline.fusion_block.parameters()),
        lr=de_cfg.hyperparameters.learning_rate
    )
    
    trainer = DecisionEngineTrainer(
        pipeline=pipeline,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=None,
        config=de_cfg,
        device=device
    )
    trainer.train()

    # 4. Load optimized weights
    best_pipeline_pth = Path(de_cfg.checkpoint_dir) / "best_pipeline.pt"
    checkpoint_pipeline = torch.load(best_pipeline_pth, map_location=device, weights_only=False)
    pipeline.load_state_dict(checkpoint_pipeline["model_state_dict"])
    pipeline = pipeline.eval()

    # 5. Dynamic Validation Profiling and Explainability Logging
    logger.info("Running explainable routing profiling on validation dataset split...")
    val_results = []
    
    # Aggregators for heatmaps
    w_records = []
    n_records = []
    b_records = []

    with torch.no_grad():
        for i in range(len(val_set)):
            # Load grayscale images and labels
            img_tensor, gt_tensor, targets = val_set[i]
            img_batch = img_tensor.unsqueeze(0).to(device)
            gt_batch = gt_tensor.unsqueeze(0).to(device)
            
            # Predict outputs
            fused, weights, order_idx, _ = pipeline(img_batch)
            
            # Individual Expert outputs for quality comparisons
            out_noise = dncnn(img_batch)
            out_sr = rcan(img_batch)
            out_struct = unet(img_batch)

            # Standardize output clamping to [0, 1] for metrics math
            fused_c = torch.clamp(fused, 0.0, 1.0)
            noise_c = torch.clamp(out_noise, 0.0, 1.0)
            sr_c = torch.clamp(out_sr, 0.0, 1.0)
            struct_c = torch.clamp(out_struct, 0.0, 1.0)
            
            # Calculate PSNRs
            fused_psnr = calculate_psnr(fused_c, gt_batch)
            noise_psnr = calculate_psnr(noise_c, gt_batch)
            sr_psnr = calculate_psnr(sr_c, gt_batch)
            struct_psnr = calculate_psnr(struct_c, gt_batch)
            
            best_expert_psnr = max(noise_psnr, sr_psnr, struct_psnr)

            # Metadata properties decoders
            noise_type_idx = int(targets["noise_type"].item())
            blur_type_idx = int(targets["blur_type"].item())
            severity_idx = int(targets["severity"].item())

            noise_type_str = REV_NOISE_MAP.get(noise_type_idx, "none")
            blur_type_str = REV_BLUR_MAP.get(blur_type_idx, "none")
            severity_str = REV_SEVERITY_MAP.get(severity_idx, "Easy")

            w_np = weights[0].cpu().numpy()
            perm = ORDER_PERMUTATIONS[order_idx]

            image_name = raw_val_set.df.iloc[i]["image_name"]

            # Save explainable entry
            record = {
                "image_name": image_name,
                "noise_type": noise_type_str,
                "noise_level": float(targets["noise_level"].item()),
                "blur_type": blur_type_str,
                "blur_strength": float(targets["blur_strength"].item()),
                "severity": severity_str,
                "confidence": float(targets["confidence"].item()),
                "expert_weights": {
                    "noise": float(w_np[0]),
                    "sr": float(w_np[1]),
                    "struct": float(w_np[2])
                },
                "processing_order": list(perm),
                "fused_psnr": fused_psnr,
                "expert_psnr_baseline": best_expert_psnr
            }
            val_results.append(record)

            # Accumulate heatmap plotting targets
            w_records.append(w_np)
            n_records.append(float(targets["noise_level"].item()))
            b_records.append(float(targets["blur_strength"].item()))

            logger.info(
                f"Image: {image_name} | Weights: N={w_np[0]:.2f}, SR={w_np[1]:.2f}, S={w_np[2]:.2f} | "
                f"Order: {'->'.join(perm)} | Fused PSNR: {fused_psnr:.2f}dB (vs {best_expert_psnr:.2f}dB Best Expert)"
            )

    # 6. Save optimization files
    # A. JSON explainable output
    json_path = reports_dir / "decision_engine_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(val_results, f, indent=4)
    logger.info(f"Saved decision routing JSON results to: {json_path.absolute()}")

    # B. CSV explainable output
    csv_path = reports_dir / "decision_engine_results.csv"
    flat_results = []
    for r in val_results:
        flat_results.append({
            "image_name": r["image_name"],
            "noise_type": r["noise_type"],
            "noise_level": r["noise_level"],
            "blur_type": r["blur_type"],
            "blur_strength": r["blur_strength"],
            "severity": r["severity"],
            "confidence": r["confidence"],
            "weight_noise": r["expert_weights"]["noise"],
            "weight_sr": r["expert_weights"]["sr"],
            "weight_struct": r["expert_weights"]["struct"],
            "processing_order": "->".join(r["processing_order"]),
            "fused_psnr": r["fused_psnr"],
            "expert_psnr_baseline": r["expert_psnr_baseline"]
        })
    df = pd.DataFrame(flat_results)
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved decision routing CSV results to: {csv_path.absolute()}")

    # C. Dynamic heatmaps plotting
    generate_decision_heatmap(reports_dir, np.array(w_records), np.array(n_records), np.array(b_records))

    # D. HTML Dashboard
    generate_html_dashboard(reports_dir, val_results)


if __name__ == "__main__":
    run_benchmarks()
