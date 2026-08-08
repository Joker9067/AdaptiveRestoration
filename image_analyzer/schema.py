"""
Schema module defining a stable, versioned prediction schema for degradation vectors (Module 7).
"""

from dataclasses import dataclass, asdict
import json
from typing import Dict, Any


@dataclass
class DegradationVector:
    """Stable, versioned degradation attribute vector estimated by Physics-Guided Image Analyzer."""
    schema_version: str
    noise_type: str
    noise_level: float
    blur_type: str
    blur_strength: float
    resolution_loss: float
    compression_quality: float
    brightness: float
    contrast: float
    gamma: float
    edge_density: float
    texture_complexity: float
    entropy: float
    severity: str
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        """Converts the degradation vector to a standard dictionary format."""
        return asdict(self)

    def to_json(self) -> str:
        """Serializes the degradation vector to a JSON string."""
        return json.dumps(self.to_dict(), indent=4)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DegradationVector":
        """Deserializes a degradation vector from a dictionary."""
        # Clean inputs to prevent structural drifts in future integration versions
        return cls(
            schema_version=data.get("schema_version", "1.0.0"),
            noise_type=str(data.get("noise_type", "none")),
            noise_level=float(data.get("noise_level", 0.0)),
            blur_type=str(data.get("blur_type", "none")),
            blur_strength=float(data.get("blur_strength", 0.0)),
            resolution_loss=float(data.get("resolution_loss", 0.0)),
            compression_quality=float(data.get("compression_quality", 100.0)),
            brightness=float(data.get("brightness", 0.0)),
            contrast=float(data.get("contrast", 0.0)),
            gamma=float(data.get("gamma", 1.0)),
            edge_density=float(data.get("edge_density", 0.0)),
            texture_complexity=float(data.get("texture_complexity", 0.0)),
            entropy=float(data.get("entropy", 0.0)),
            severity=str(data.get("severity", "easy")),
            confidence=float(data.get("confidence", 1.0))
        )
