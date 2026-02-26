"""Provides transformations for calculating gaze dispersion.

This module contains calculators for standard gaze dispersion and for
dispersion relative to the motion of the background (optical flow).
"""

from functools import partial

import numpy as np
import pandas as pd

from datasets import GazeDataset

from .transformation import Transformation
from .utils import add_flow_velocity


class DispersionCalculator(Transformation):
    """Calculates gaze dispersion over a sliding window.

    Dispersion is defined as the sum of the range (max - min) of x and y
    gaze coordinates within a temporal window.

    Args:
        x_col: The name of the column containing x-coordinates.
        y_col: The name of the column containing y-coordinates.
        time_col: The name of the column containing timestamps.

    """

    def __init__(self, x_col: str = "x", y_col: str = "y", time_col: str = "timestamp"):
        self.x_col_name = x_col
        self.y_col_name = y_col
        self.t_col_name = time_col

    def apply(self, data: GazeDataset):
        """Apply the dispersion calculation to all splits in the dataset.

        Args:
            data: The `GazeDataset` to be transformed.

        """
        for _key, split in data.get_splits():
            window_size = int(25 / data.sample_duration_ms)

            x_spread = (
                split[self.x_col_name]
                .rolling(window_size, center=True, min_periods=1)
                .apply(
                    lambda x: x.max() - x.min(),
                )
            )
            y_spread = (
                split[self.y_col_name]
                .rolling(window_size, center=True, min_periods=1)
                .apply(
                    lambda y: y.max() - y.min(),
                )
            )
            split["dispersion"] = x_spread + y_spread


class RelativeDispersionCalculator(DispersionCalculator):
    """Calculates gaze dispersion relative to optical flow.

    This calculator first determines an "ideal" gaze trajectory by integrating
    the optical flow within a temporal window. It then calculates the dispersion
    of the actual gaze points relative to this ideal trajectory.
    """

    def __init__(self, x_col: str = "x", y_col: str = "y", time_col: str = "timestamp"):
        super().__init__(x_col, y_col, time_col)

    def apply(self, data: GazeDataset):
        """Apply the relative dispersion calculation to the dataset.

        Args:
            data: The `GazeDataset` to be transformed.

        """

        def caculate_window_dispersion(series: np.ndarray, split: pd.DataFrame) -> float:
            window_indices = series.astype(int)
            window = split.iloc[window_indices]

            center_pos = len(window) // 2

            center_x = window[self.x_col_name].iloc[center_pos]
            center_y = window[self.y_col_name].iloc[center_pos]

            dx = window["flow_x_vel"].to_numpy() * data.sample_duration_ms / 1000
            dy = window["flow_y_vel"].to_numpy() * data.sample_duration_ms / 1000

            traj_x = np.zeros_like(dx)
            traj_y = np.zeros_like(dy)

            # FORWARD: Integrate from center+1 to end
            traj_x[center_pos + 1 :] = np.cumsum(dx[center_pos + 1 :])
            traj_y[center_pos + 1 :] = np.cumsum(dy[center_pos + 1 :])

            # BACKWARD: Integrate from center-1 to start
            if center_pos > 0:
                traj_x[:center_pos] = -np.cumsum(dx[:center_pos][::-1])[::-1]
                traj_y[:center_pos] = -np.cumsum(dy[:center_pos][::-1])[::-1]

            # Center is left as 0.0 (anchored)

            ideal_x = center_x + traj_x
            ideal_y = center_y + traj_y

            rel_x = window[self.x_col_name].to_numpy() - ideal_x
            rel_y = window[self.y_col_name].to_numpy() - ideal_y

            return (np.max(rel_x) - np.min(rel_x)) + (np.max(rel_y) - np.min(rel_y))

        super().apply(data)
        window_size = int(25 / data.sample_duration_ms)
        for _key, split in data.get_splits():
            add_flow_velocity(split)

            indexer = pd.Series(np.arange(len(split)), index=split.index)
            split["rel_dispersion"] = indexer.rolling(
                window_size,
                center=True,
                min_periods=1,
            ).apply(partial(caculate_window_dispersion, split=split), raw=True)
