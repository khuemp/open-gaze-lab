"""Provides utility functions for data transformations."""

import numpy as np
import pandas as pd


def add_flow_velocity(split: pd.DataFrame) -> None:
    """Calculate and add optical flow velocity columns to a DataFrame.

    This function computes the x and y components of the optical flow velocity
    based on the 'flow_x', 'flow_y', and 'video_timestamp' columns. It also
    calculates and adds the magnitude of the flow velocity vector.

    Note:
        The calculation uses the time delta between consecutive frames, which
        is considered more accurate for this purpose than using an average
        sample rate.

    Args:
        split: The DataFrame to which flow velocity columns will be added.
               It must contain 'flow_x', 'flow_y', and 'video_timestamp'.

    """
    # NOTE: In contrast to gaze velocity calculation, using actual time deltas instead
    # of the average is essential in achieving better performance for relative algorithms
    split["flow_t_delta"] = split["video_timestamp"].diff(1)
    split["flow_t_delta"] = split["flow_t_delta"].replace(0, np.nan).ffill().fillna(0)

    split["flow_x_vel"] = split["flow_x"] / split["flow_t_delta"]
    split["flow_y_vel"] = split["flow_y"] / split["flow_t_delta"]

    # NOTE: Interpolating instead of forward filling does not improve performance
    # split["flow_x_vel2"] = split["flow_x"] / split["flow_t_delta"]
    # split["flow_y_vel2"] = split["flow_y"] / split["flow_t_delta"]

    # split["flow_x_vel"] = np.nan
    # split["flow_y_vel"] = np.nan
    # frames, funique = np.unique(split["frame"], return_index=True)
    # split.loc[funique, "flow_x_vel"] = split.loc[funique, "flow_x_vel2"]
    # split.loc[funique, "flow_y_vel"] = split.loc[funique, "flow_y_vel2"]
    # split.reset_index(drop=True, inplace=True)
    # # We have to remove infinite values because methods like "spline" won't work otherwise
    # split.loc[split["flow_x_vel"].isin([np.inf, -np.inf]), "flow_x_vel"] = np.nan
    # split.loc[split["flow_y_vel"].isin([np.inf, -np.inf]), "flow_y_vel"] = np.nan
    # split["flow_x_vel"] = split["flow_x_vel"].interpolate(method="cubic")
    # split["flow_y_vel"] = split["flow_y_vel"].interpolate(method="cubic")

    split["flow_vel_mag"] = np.hypot(split["flow_x_vel"], split["flow_y_vel"])
