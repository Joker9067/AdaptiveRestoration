# U-Net: Grayscale Structure Restoration Model

This directory implements the classic **U-Net** encoder-decoder architecture with skip connections for restoration.

## Architecture

- **Encoder**: Contracting path with down-sampling Conv-ReLU blocks.
- **Decoder**: Expanding path with ConvTranspose2d up-sampling layers and skip connection concatenations from the contracting path.
- **Grayscale Optimization**: Standardized for single-channel (1 input channel, 1 output channel) images.

## Configurations

Configurable settings are stored in `config.py` under `DEFAULT_CONFIG`.
- `optimizer`: `Adam`, `AdamW`, or `SGD`
- `lr_scheduler`: `CosineAnnealingLR` or `MultiStepLR`
- `loss_weights`: Combined `l1` + `ssim` loss.
