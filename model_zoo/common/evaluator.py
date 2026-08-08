"""
Evaluator and benchmarking module for the Semiconductor Image Restoration System (Module 6).
Evaluates models against PSNR, SSIM, LPIPS, inference latency, parameter count,
FLOPs, and maximum GPU memory allocation.
"""

import time
import logging
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from model_zoo.common.metrics import calculate_psnr, calculate_ssim, calculate_lpips

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Orchestrates comprehensive metrics benchmarking for Zoo Models."""

    def __init__(self, model: nn.Module, test_loader: DataLoader, device: torch.device) -> None:
        """Initializes the evaluator.

        Args:
            model (nn.Module): The model to benchmark.
            test_loader (DataLoader): DataLoader for evaluation.
            device (torch.device): Device configuration.
        """
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device

    def evaluate(self, input_size: Tuple[int, int, int, int] = (1, 1, 512, 512)) -> Dict[str, Any]:
        """Runs the complete evaluation suite on the dataset.

        Args:
            input_size (Tuple[int, int, int, int]): The size of a single sample tensor.

        Returns:
            Dict[str, Any]: Metric logs.
        """
        self.model.eval()

        # 1. Complexities (Parameters and FLOPs)
        # Check if the model is wrapped in DataParallel
        unwrapped_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        param_count = unwrapped_model.get_parameter_count()
        flops = unwrapped_model.get_flops(input_size=input_size)

        # Warmup pass
        dummy = torch.zeros(input_size, device=self.device)
        try:
            with torch.no_grad():
                for _ in range(5):
                    self.model(dummy)
        except Exception:
            pass

        # 2. Reset CUDA Memory stats if active
        is_cuda = self.device.type == "cuda"
        if is_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.empty_cache()

        # 3. Validation Accumulators
        psnr_acc = 0.0
        ssim_acc = 0.0
        lpips_acc = 0.0
        total_samples = 0
        
        # Benchmark inference latency
        latency_records = []

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                batch_size = inputs.size(0)
                total_samples += batch_size

                # Time single inference step
                t_start = time.perf_counter()
                outputs = self.model(inputs)
                
                # Enforce CUDA sync if active for accurate profiling
                if is_cuda:
                    torch.cuda.synchronize(self.device)
                t_duration = time.perf_counter() - t_start
                latency_records.append(t_duration / batch_size)

                # Clamp outputs to valid range
                outputs_clamped = torch.clamp(outputs, 0.0, 1.0)
                
                # Compute metrics
                psnr_acc += calculate_psnr(outputs_clamped, targets) * batch_size
                ssim_acc += calculate_ssim(outputs_clamped, targets) * batch_size
                lpips_acc += calculate_lpips(outputs_clamped, targets) * batch_size

        # 4. Memory Profiling
        gpu_mem_mb = 0.0
        if is_cuda:
            gpu_mem_mb = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)

        # 5. Formulate averages
        avg_psnr = psnr_acc / total_samples
        avg_ssim = ssim_acc / total_samples
        avg_lpips = lpips_acc / total_samples
        avg_latency_ms = np.mean(latency_records) * 1000.0

        return {
            "psnr": float(avg_psnr),
            "ssim": float(avg_ssim),
            "lpips": float(avg_lpips),
            "inference_time_ms": float(avg_latency_ms),
            "gpu_memory_mb": float(gpu_mem_mb),
            "parameter_count": int(param_count),
            "flops": int(flops)
        }
