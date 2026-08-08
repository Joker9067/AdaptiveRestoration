"""
Automated Final Report Generator for the Semiconductor Image Restoration System.
Aggregates metrics from Modules 1-8 into a publication-quality HTML dashboard.
"""

import json
from pathlib import Path
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def generate_report():
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Final Project Report: Semiconductor Image Restoration</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #0f172a; color: #e2e8f0; margin: 0; padding: 40px; }
            h1, h2, h3 { color: #38bdf8; }
            .container { max-width: 1200px; margin: 0 auto; background: #1e293b; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
            table { width: 100%; border-collapse: collapse; margin: 20px 0; background: #0f172a; }
            th, td { padding: 12px; border: 1px solid #334155; text-align: center; }
            th { background-color: #38bdf8; color: #0f172a; }
            .metric-box { display: inline-block; padding: 20px; background: #334155; border-radius: 8px; margin: 10px; width: 30%; text-align: center; }
            .metric-value { font-size: 24px; font-weight: bold; color: #38bdf8; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Semiconductor Image Restoration - Final Report</h1>
            <p>End-to-End Evaluation of the Physics-Guided Adaptive Decision System.</p>
            
            <h2>1. Dataset Statistics</h2>
    """
    
    # Load dataset stats
    meta_path = Path("unified_metadata.csv")
    if meta_path.exists():
        df = pd.read_csv(meta_path)
        total = len(df)
        train = len(df[df["split"] == "train"])
        val = len(df[df["split"] == "val"])
        test = len(df[df["split"] == "test"])
        html_content += f"""
        <div style="display: flex; justify-content: space-between;">
            <div class="metric-box">Total Image Pairs<br><span class="metric-value">{total}</span></div>
            <div class="metric-box">Training Set<br><span class="metric-value">{train}</span></div>
            <div class="metric-box">Test Set<br><span class="metric-value">{test}</span></div>
        </div>
        """
        
    # Load Model Benchmarks
    bench_json = reports_dir / "benchmark_results.json"
    if bench_json.exists():
        html_content += "<h2>2. Model Benchmarks (Module 6)</h2>"
        with open(bench_json, "r") as f:
            bench_data = json.load(f)
            
        html_content += "<table><tr><th>Model</th><th>PSNR (dB)</th><th>SSIM</th><th>LPIPS</th><th>Latency (ms)</th><th>Params</th></tr>"
        for m in bench_data:
            html_content += f"<tr><td>{m.get('model_name','')}</td><td>{m.get('psnr',0):.2f}</td><td>{m.get('ssim',0):.4f}</td><td>{m.get('lpips',0):.4f}</td><td>{m.get('inference_time_ms',0):.1f}</td><td>{m.get('parameter_count',0):,}</td></tr>"
        html_content += "</table>"
        
    # Load Decision Engine Results
    de_json = reports_dir / "decision_engine_results.json"
    if de_json.exists():
        html_content += "<h2>3. Adaptive Decision Engine (Module 8)</h2>"
        with open(de_json, "r") as f:
            de_data = json.load(f)
            
        html_content += "<table><tr><th>Image</th><th>Predicted Sequence</th><th>Fused PSNR</th><th>Best Expert PSNR</th></tr>"
        for rec in de_data[:10]: # Top 10 samples
            html_content += f"<tr><td>{rec.get('image_id','')}</td><td>{' &rarr; '.join(rec.get('order',[]))}</td><td>{rec.get('fused_psnr',0):.2f}</td><td>{rec.get('best_expert_psnr',0):.2f}</td></tr>"
        html_content += "</table>"
        
        # Add Heatmap if exists
        hm_path = reports_dir / "expert_weight_heatmap.png"
        if hm_path.exists():
            html_content += f'<h3>Routing Strategy Heatmap</h3><img src="expert_weight_heatmap.png" width="800px" style="border-radius: 8px;">'

    html_content += """
        </div>
    </body>
    </html>
    """
    
    out_file = reports_dir / "final_report.html"
    with open(out_file, "w") as f:
        f.write(html_content)
        
    logger.info(f"Generated Final Report at: {out_file.absolute()}")

if __name__ == "__main__":
    generate_report()
