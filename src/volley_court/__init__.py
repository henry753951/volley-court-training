"""Fast, typed volleyball court-line inference and training."""

from .api import CourtLineModel, InferenceConfig
from .assets import DEFAULT_MODEL, ModelIntegrityError, ModelSpec, download_model
from .tracking import CourtLayoutTracker, LayoutTrackingConfig
from .types import CourtFrameResult, CourtKeypoint, CourtLayout, CourtLine
from .visualization import CourtVisualizer, VisualizationConfig

__version__ = "0.2.0"

__all__ = [
    "DEFAULT_MODEL",
    "CourtFrameResult",
    "CourtKeypoint",
    "CourtLayout",
    "CourtLayoutTracker",
    "CourtLine",
    "CourtLineModel",
    "CourtVisualizer",
    "InferenceConfig",
    "LayoutTrackingConfig",
    "ModelIntegrityError",
    "ModelSpec",
    "VisualizationConfig",
    "__version__",
    "download_model",
]
