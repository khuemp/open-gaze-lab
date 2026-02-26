"""Provides a collection of data transformations for eye-tracking datasets.

These transformations are designed to be applied to `GazeDataset` objects,
adding new feature columns or modifying existing ones in-place. They range
from simple filters (Savgol) to complex feature calculations (optical flow,
entropy, etc.).
"""

from .adaptive_threshold import AdaptiveThreshold
from .dispersion_calculator import DispersionCalculator, RelativeDispersionCalculator
from .entropy_calculator import EntropyCalculator
from .optic_flow_lk import OpticFlowLK
from .optic_flow_loader import OpticFlowLoader
from .patch_similarity_loader_drews import PatchSimilarityLoaderDrews
from .patch_similarity_predictor import PatchSimilarityPredictor
from .savgol_filter import SavgolFilter
from .transformation import Transformation
from .velocity_calculator import RelativeVelocityCalculator, VelocityCalculator
from .vo_loader import VOLoader

__all__ = [
    "AdaptiveThreshold",
    "DispersionCalculator",
    "EntropyCalculator",
    "OpticFlowLK",
    "OpticFlowLoader",
    "PatchSimilarityLoaderDrews",
    "PatchSimilarityPredictor",
    "RelativeDispersionCalculator",
    "RelativeVelocityCalculator",
    "SavgolFilter",
    "Transformation",
    "VOLoader",
    "VelocityCalculator",
]
