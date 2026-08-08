"""
Inference script for Ridnet model.
Loads a trained checkpoint and restores a noisy input image.
"""
import os
import argparse
import logging
from pathlib import Path
import cv2
import numpy as np
import torch

from model_zoo.ridnet.model import Ridnet
from model_zoo.common.metrics import calculate_psnr, calculate_ssim

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_inference(input_path: str, output_path: str, checkpoint_path: str, gt_path: str = None) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using inference device: {device}")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input image not found: {input_path}")
    
    noisy_img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    
    input_tensor = torch.from_numpy(noisy_img).float().unsqueeze(0).unsqueeze(0) / 255.0
    input_tensor = input_tensor.to(device)

    model = Ridnet()
    if not os.path.exists(checkpoint_path):
        logger.warning(f"No checkpoint found at {checkpoint_path}. Running with random initialization.")
    else:
        logger.info(f"Loading weights from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        output_tensor = model(input_tensor)
        output_clamped = torch.clamp(output_tensor, 0.0, 1.0)
    
    restored_np = (output_clamped.squeeze().cpu().numpy() * 255.0).astype(np.uint8)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, restored_np)
    logger.info(f"Saved restored image output to {output_path}")

    if gt_path and os.path.exists(gt_path):
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        gt_tensor = torch.from_numpy(gt_img).float().unsqueeze(0).unsqueeze(0) / 255.0
        gt_tensor = gt_tensor.to(device)

        psnr_val = calculate_psnr(output_clamped, gt_tensor)
        ssim_val = calculate_ssim(output_clamped, gt_tensor)
        logger.info(f"Inference metrics evaluation - PSNR: {psnr_val:.2f}dB | SSIM: {ssim_val:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ridnet restoration inference script")
    parser.add_argument("--input", type=str, required=True, help="Noisy input image path")
    parser.add_argument("--output", type=str, required=True, help="Restored output image destination path")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/ridnet/best.pt", help="Checkpoint model weights path")
    parser.add_argument("--gt", type=str, default=None, help="Ground truth image path for metrics verification")
    args = parser.parse_args()

    run_inference(args.input, args.output, args.checkpoint, args.gt)
