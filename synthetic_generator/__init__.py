"""
Synthetic Generator Module for Semiconductor Image Restoration.
Exposes core classes for generating degraded images, presets, registries, and degradation operators.
"""

from synthetic_generator.degradations import (
    BaseDegradation,
    DegradationRegistry,
)
from synthetic_generator.pipeline import (
    DegradationPipeline,
    PRESETS,
)
from synthetic_generator.generator import (
    SyntheticDatasetGenerator,
    create_mock_pattern,
)
