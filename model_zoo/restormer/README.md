# Restormer: Grayscale Image Restoration Model

This directory implements the **Restormer** architecture.

## Architecture

Standardized for single-channel (1 input channel, 1 output channel) images.

## Configurations

Configurable settings are stored in `config.py` under `DEFAULT_CONFIG`.
- `optimizer`: `Adam`, `AdamW`, or `SGD`
- `lr_scheduler`: `CosineAnnealingLR` or `MultiStepLR`
- `loss_weights`: Combined loss mapping (L1 + SSIM).
