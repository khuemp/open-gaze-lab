"""Implements our two-threshold event classification algorithm."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from transformations import (
    AdaptiveThreshold,
    RelativeVelocityCalculator,
    SavgolFilter,
    Transformation,
    VelocityCalculator,
)

from algorithms import Algorithm, IVTParams
from datasets import GazeDataset
from event_matching.event import EventType


@dataclass(frozen=True)
class TwoThresholdParams(IVTParams):
    """Dataclass for the Two-Threshold algorithm parameters.

    Inherits from `IVTParams` and adds a second threshold to distinguish
    pursuit movements.

    Attributes:
        sp_threshold: The velocity threshold for separating Saccade and Pursuit.

    """

    sp_threshold: int = 1300


class TwoThreshold(Algorithm[TwoThresholdParams]):
    """An extension of the I-VT algorithm that adds a second threshold.

    This algorithm distinguishes between three event types: Fixation, Pursuit,
    and Saccade. It uses a lower velocity threshold to separate fixations from
    moving events, and a higher threshold to separate pursuits from saccades.
    """

    def __init__(self, data: GazeDataset, *, use_flow: bool):
        """Initialise the Two-Threshold algorithm.

        Args:
            data: The `GazeDataset` to be processed.
            use_flow: If True, uses relative velocity (gaze vs. flow). If
                      False, uses standard gaze velocity.

        """
        self.use_flow = use_flow
        preprocessing: list[Transformation] = [SavgolFilter(55, data.sample_rate_hz)]

        if self.use_flow:
            preprocessing.append(RelativeVelocityCalculator(x_col="filter_x", y_col="filter_y"))
        else:
            preprocessing.append(VelocityCalculator(x_col="filter_x", y_col="filter_y"))

        super().__init__(data, preprocessing)

    def classify(self, params: TwoThresholdParams) -> None:
        """Classify events using two velocity thresholds.

        Classifies samples as Fixation, Pursuit, or Saccade based on where the
        velocity magnitude falls relative to the two thresholds. Supports both
        fixed and adaptive thresholds.

        Args:
            params: The parameters for the classification.

        """
        self.matchers = []
        mag_name = "vel_rel_mag" if self.use_flow else "vel_mag"

        if params.gain is None:
            assert params.window_size_ms is None
            for _key, split in self.data.get_splits():
                split["prediction"] = pd.cut(
                    split[mag_name],
                    [-np.inf, params.threshold, params.sp_threshold, np.inf],
                    labels=[EventType.FIXATION, EventType.PURSUIT, EventType.SACCADE],
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
            process2 = AdaptiveThreshold(
                window_size,
                params.gain,
                params.sp_threshold,
                "sp_threshold",
            )
            process2.apply(self.data)

            for _key, split in self.data.get_splits():
                conditions = [
                    split[mag_name] < split["threshold"],
                    (split["threshold"] <= split[mag_name])
                    & (split[mag_name] < split["sp_threshold"]),
                ]
                choices = [EventType.FIXATION, EventType.PURSUIT]
                split["prediction"] = np.select(conditions, choices, default=EventType.SACCADE)
