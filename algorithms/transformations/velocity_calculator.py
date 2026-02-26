"""Provides transformations for calculating gaze velocity.

This module contains calculators for standard gaze velocity and for velocity
relative to the motion of the background (optical flow).
"""

import numpy as np

from datasets import GazeDataset

from .transformation import Transformation
from .utils import add_flow_velocity


class VelocityCalculator(Transformation):
    """Calculates gaze velocity from coordinate and timestamp data.

    This transformation computes the velocity and direction of gaze movements
    based on the differences in x/y coordinates and timestamps between
    consecutive samples.

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
        """Apply the velocity calculation to all splits in the dataset.

        Args:
            data: The `GazeDataset` to be transformed.

        """
        for _key, split in data.get_splits():
            split["x_delta"] = -split[self.x_col_name].diff(-1)
            split["y_delta"] = -split[self.y_col_name].diff(-1)
            split["t_delta"] = -split[self.t_col_name].diff(-1)
            split.fillna(0, inplace=True)
            avg_delta = split["t_delta"].mean()
            split["x_vel"] = split["x_delta"] / avg_delta
            split["y_vel"] = split["y_delta"] / avg_delta
            split["vel_mag"] = np.hypot(split["x_vel"], split["y_vel"])
            split["direction"] = np.arctan2(-split["y_delta"], -split["x_delta"])


class RelativeVelocityCalculator(VelocityCalculator):
    """Calculates gaze velocity relative to the underlying optical flow.

    This transformation first calculates the standard gaze velocity and then
    computes the velocity of the gaze relative to the optical flow at the gaze
    point. It also calculates the cosine similarity between the gaze and flow
    vectors.

    """

    def __init__(self, x_col: str = "x", y_col: str = "y", time_col: str = "timestamp"):
        super().__init__(x_col, y_col, time_col)

    def apply(self, data: GazeDataset):
        """Apply the relative velocity calculation to the dataset.

        Args:
            data: The `GazeDataset` to be transformed.

        """
        super().apply(data)
        for _key, split in data.get_splits():
            add_flow_velocity(split)

            split["x_vel_rel"] = split["x_vel"] - split["flow_x_vel"]
            split["y_vel_rel"] = split["y_vel"] - split["flow_y_vel"]

            split["vel_rel_mag"] = np.hypot(split["x_vel_rel"], split["y_vel_rel"])

            numerator = split["x_vel"] * split["flow_x_vel"] + split["y_vel"] * split["flow_y_vel"]
            denominator = split["vel_mag"] * split["flow_vel_mag"]

            split["cosine_sim"] = numerator / denominator
