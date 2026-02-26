"""Implements the I-VT (Velocity Threshold) event classification algorithm."""

from dataclasses import dataclass

import numpy as np
from transformations import (
    AdaptiveThreshold,
    RelativeVelocityCalculator,
    SavgolFilter,
    Transformation,
    VelocityCalculator,
)

from algorithms import Algorithm, AlgorithmParams
from datasets import GazeDataset
from event_matching.event import EventType


@dataclass(frozen=True)
class IVTParams(AlgorithmParams):
    """Dataclass for I-VT algorithm parameters.

    Attributes:
        threshold: The velocity threshold for classifying fixations.
        gain: The gain factor for the adaptive threshold calculation. If None,
              a fixed threshold is used.
        window_size_ms: The window size in milliseconds for the adaptive
                        threshold calculation.

    """

    threshold: int = 1200
    gain: float | None = None
    window_size_ms: int | None = None


class IVT(Algorithm[IVTParams]):
    """An implementation of the I-VT (Velocity Threshold) algorithm.

    This algorithm classifies gaze samples into fixations or saccades based on
    a velocity threshold. It can operate in two modes:
    1.  Standard I-VT: Uses gaze velocity.
    2.  Flow-assisted I-VT: Uses relative velocity between gaze and optical flow.

    It also supports both a fixed threshold and an adaptive threshold that
    adjusts based on the root mean square of optical flow in a local window.
    """

    def __init__(self, data: GazeDataset, *, use_flow: bool):
        """Initialise the I-VT algorithm.

        Args:
            data: The `GazeDataset` to be processed.
            use_flow: If True, uses relative velocity (gaze vs. flow). If
                      False, uses standard gaze velocity.

        """
        self.use_flow = use_flow
        preprocessing: list[Transformation] = [SavgolFilter(55, data.sample_rate_hz)]

        if self.use_flow:
            # preprocessing.append(OpticFlowLoader())
            # preprocessing.append(OpticFlowLK())
            preprocessing.append(RelativeVelocityCalculator(x_col="filter_x", y_col="filter_y"))
        else:
            preprocessing.append(VelocityCalculator(x_col="filter_x", y_col="filter_y"))

        super().__init__(data, preprocessing)

    def classify(self, params: IVTParams) -> None:
        """Classify events as Fixation or Saccade based on a velocity threshold.

        If `params.gain` is provided, an adaptive threshold is calculated by
        adding a scaled optical flow RMS value to the base threshold.
        Otherwise, a fixed threshold is used.

        Args:
            params: The parameters for the I-VT classification.
        """
        self.matchers = []
        mag_name = "vel_rel_mag" if self.use_flow else "vel_mag"

        if params.gain is None:
            assert params.window_size_ms is None
            for _key, split in self.data.get_splits():
                split["prediction"] = np.where(
                    split[mag_name] < params.threshold,
                    EventType.FIXATION,
                    EventType.SACCADE,
                )
        else:
            assert self.use_flow, "Using adaptive thresholding requires optical flow calculation"
            assert params.gain is not None
            assert params.window_size_ms is not None
            assert params.window_size_ms > 0
            assert params.gain >= 0

            window_size = int(params.window_size_ms // self.data.sample_duration_ms)
            process = AdaptiveThreshold(window_size, params.gain, params.threshold)
            process.apply(self.data)

            for _key, split in self.data.get_splits():
                split["prediction"] = np.where(
                    split[mag_name] < split["threshold"],
                    EventType.FIXATION,
                    EventType.SACCADE,
                )
