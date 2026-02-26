"""Provides transformations for calculating adaptive thresholds.

This module contains classes for calculating the Root Mean Square (RMS) of
optical flow within a sliding window and for using that value to compute an
adaptive threshold for event classification.
"""

import numpy as np

from datasets import GazeDataset

from .transformation import Transformation


class FlowWindowRMS(Transformation):
    """Calculates the RMS of optical flow within a sliding window.

    This transformation computes the Root Mean Square of the x and y components
    of optical flow velocity over a centered rolling window. It then adds a
    column for the magnitude of this RMS vector to the dataset splits.

    Args:
        window_size: The size of the sliding window in number of samples.

    """

    def __init__(self, window_size: int):
        self.window_size = window_size

    def apply(self, data: GazeDataset):
        """Apply the RMS calculation to all splits in the dataset.

        Args:
            data: The `GazeDataset` to be transformed.

        """

        def rms(x: np.ndarray) -> np.ndarray:
            return np.sqrt(np.mean(x**2))

        for _key, split in data.get_splits():
            assert "flow_x_vel" in split.columns
            assert "flow_y_vel" in split.columns

            x_col_name = f"flow_rms_x_{self.window_size}"
            y_col_name = f"flow_rms_y_{self.window_size}"
            mag_col_name = f"flow_rms_mag_{self.window_size}"

            if mag_col_name in split.columns:
                return

            # TODO: probably don't need to save these 2 in the dataframe
            #   test if leaving away makes significant difference
            split[x_col_name] = (
                split["flow_x_vel"]
                .rolling(window=self.window_size, center=True, min_periods=1)
                .apply(rms, raw=True)
            )
            split[y_col_name] = (
                split["flow_y_vel"]
                .rolling(window=self.window_size, center=True, min_periods=1)
                .apply(rms, raw=True)
            )
            split[mag_col_name] = np.hypot(split[x_col_name], split[y_col_name])


class AdaptiveThreshold(Transformation):
    """Calculates an adaptive threshold based on local optical flow.

    This transformation computes a dynamic threshold by adding a scaled (gain)
    local optical flow RMS value to a fixed base threshold. It uses
    `FlowWindowRMS` as a prerequisite calculation.

    Args:
        window_size: The size of the sliding window for the RMS calculation.
        gain: The gain factor to scale the flow RMS value.
        base_threshold: The fixed base value of the threshold.
        result_col: The name of the column to store the final threshold.

    """

    def __init__(
        self,
        window_size: int,
        gain: float,
        base_threshold: int,
        result_col: str = "threshold",
    ):
        self.window_size = window_size
        self.gain = gain
        self.base_threshold = base_threshold
        self.flow_window_rms = FlowWindowRMS(self.window_size)
        self.result_col = result_col

    def apply(self, data: GazeDataset):
        """Apply the adaptive threshold calculation to the dataset.

        Args:
            data: The `GazeDataset` to be transformed.

        """
        self.flow_window_rms.apply(data)
        for _key, split in data.get_splits():
            split[self.result_col] = (
                self.base_threshold + self.gain * split[f"flow_rms_mag_{self.window_size}"]
            )
