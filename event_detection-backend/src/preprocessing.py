"""Preprocessing transformations for enhanced eye-tracking event detection.

Ports key preprocessing steps from the research I-VAT+Frel algorithm
(Savitzky-Golay filtering, velocity calculation, optical flow compensation,
adaptive thresholding) to work on plain DataFrames within the web backend.

Based on: "Strategies for enhancing automatic fixation detection in
head-mounted eye tracking" — the I-VAT+Frel variant (Adaptive Velocity
Threshold with Head-Motion Compensation).
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


# ---------------------------------------------------------------------------
# Column detection helpers
# ---------------------------------------------------------------------------

def detect_flow_columns(df: pd.DataFrame) -> bool:
    """Check whether the DataFrame contains optical-flow displacement columns.

    The research pipeline expects ``flow_x`` and ``flow_y`` columns that store
    the per-sample (or per-frame) optical-flow displacement at the gaze point.

    Returns:
        True if both ``flow_x`` and ``flow_y`` are present.
    """
    return "flow_x" in df.columns and "flow_y" in df.columns


def detect_video_timestamp_column(df: pd.DataFrame) -> bool:
    """Check whether a ``video_timestamp`` column exists."""
    return "video_timestamp" in df.columns


# ---------------------------------------------------------------------------
# Savitzky-Golay smoothing  (mirrors algorithms/transformations/savgol_filter.py)
# ---------------------------------------------------------------------------

def apply_savgol_filter(
    df: pd.DataFrame,
    sampling_rate: float,
    window_size_ms: float = 55,
    polyorder: int = 3,
) -> pd.DataFrame:
    """Apply a Savitzky-Golay low-pass filter to gaze coordinates.

    Adds ``filter_x`` and ``filter_y`` columns to *df* (in-place).

    Args:
        df: DataFrame with ``x`` and ``y`` columns.
        sampling_rate: Recording sampling rate in Hz.
        window_size_ms: Filter window size in milliseconds.
        polyorder: Polynomial order for the Savitzky-Golay filter.

    Returns:
        The same DataFrame with added ``filter_x`` / ``filter_y`` columns.
    """
    frame_duration_ms = 1000.0 / sampling_rate
    window_size = int(window_size_ms // frame_duration_ms)
    # Window must be odd and > polyorder
    if window_size < polyorder + 2:
        window_size = polyorder + 2
    if window_size % 2 == 0:
        window_size += 1

    # Fill NaN temporarily for filtering (restore later is unnecessary — the
    # downstream blink detector will still flag those rows via the original
    # x / y columns).
    x_filled = df["x"].fillna(0).values
    y_filled = df["y"].fillna(0).values

    if len(x_filled) < window_size:
        # Not enough data for the requested window — fall back to raw coords
        df["filter_x"] = df["x"].values.copy()
        df["filter_y"] = df["y"].values.copy()
    else:
        df["filter_x"] = savgol_filter(x_filled, window_length=window_size, polyorder=polyorder)
        df["filter_y"] = savgol_filter(y_filled, window_length=window_size, polyorder=polyorder)

    return df


# ---------------------------------------------------------------------------
# Gaze velocity  (mirrors algorithms/transformations/velocity_calculator.py)
# ---------------------------------------------------------------------------

def compute_gaze_velocity(
    df: pd.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
    time_col: str = "timestamp",
) -> pd.DataFrame:
    """Compute gaze velocity using the *average* time delta.

    This mirrors the research code's ``VelocityCalculator`` which divides
    spatial deltas by the mean Δt across the recording (rather than per-sample
    Δt).  Using a constant denominator reduces noise in the velocity signal
    when the sampling interval varies slightly.

    Adds columns: ``x_vel``, ``y_vel``, ``vel_mag``.

    Returns:
        The same DataFrame with added velocity columns.
    """
    x_delta = -df[x_col].diff(-1).fillna(0)
    y_delta = -df[y_col].diff(-1).fillna(0)
    t_delta = -df[time_col].diff(-1).fillna(0)

    avg_delta = t_delta.mean()
    if avg_delta == 0 or np.isnan(avg_delta):
        avg_delta = 1.0  # safety

    df["x_vel"] = x_delta / avg_delta
    df["y_vel"] = y_delta / avg_delta
    df["vel_mag"] = np.hypot(df["x_vel"], df["y_vel"])

    return df


# ---------------------------------------------------------------------------
# Optical-flow velocity  (mirrors algorithms/transformations/utils.py  add_flow_velocity)
# ---------------------------------------------------------------------------

def compute_flow_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """Convert per-frame optical-flow displacements to velocities.

    Expects ``flow_x``, ``flow_y`` columns (displacement in pixels) and a
    time reference column.  If ``video_timestamp`` exists it is used; otherwise
    the regular ``timestamp`` column provides the time deltas.

    Adds columns: ``flow_x_vel``, ``flow_y_vel``, ``flow_vel_mag``.

    Returns:
        The same DataFrame with added flow velocity columns.
    """
    time_col = "video_timestamp" if "video_timestamp" in df.columns else "timestamp"

    flow_t_delta = df[time_col].diff(1)
    flow_t_delta = flow_t_delta.replace(0, np.nan).ffill().bfill().fillna(1)

    df["flow_x_vel"] = df["flow_x"] / flow_t_delta
    df["flow_y_vel"] = df["flow_y"] / flow_t_delta

    # Replace inf values from potential zero-division
    df["flow_x_vel"] = df["flow_x_vel"].replace([np.inf, -np.inf], 0)
    df["flow_y_vel"] = df["flow_y_vel"].replace([np.inf, -np.inf], 0)

    df["flow_vel_mag"] = np.hypot(df["flow_x_vel"], df["flow_y_vel"])

    return df


# ---------------------------------------------------------------------------
# Relative velocity  (mirrors RelativeVelocityCalculator)
# ---------------------------------------------------------------------------

def compute_relative_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """Subtract optical-flow velocity from gaze velocity.

    This is the core of the *Frel* (flow-relative) head-motion compensation.
    Requires ``x_vel``, ``y_vel`` (from ``compute_gaze_velocity``) and
    ``flow_x_vel``, ``flow_y_vel`` (from ``compute_flow_velocity``).

    Adds columns: ``x_vel_rel``, ``y_vel_rel``, ``vel_rel_mag``.

    Returns:
        The same DataFrame with added relative-velocity columns.
    """
    df["x_vel_rel"] = df["x_vel"] - df["flow_x_vel"]
    df["y_vel_rel"] = df["y_vel"] - df["flow_y_vel"]
    df["vel_rel_mag"] = np.hypot(df["x_vel_rel"], df["y_vel_rel"])

    return df


# ---------------------------------------------------------------------------
# Flow-window RMS  (mirrors FlowWindowRMS)
# ---------------------------------------------------------------------------

def compute_flow_window_rms(df: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """Compute the rolling RMS of optical-flow velocity.

    The RMS magnitude is used as an indicator of how much the background is
    moving in a local temporal neighbourhood.

    Adds column: ``flow_rms_mag``.

    Returns:
        The same DataFrame with added ``flow_rms_mag`` column.
    """

    def _rms(x: np.ndarray) -> float:
        return np.sqrt(np.mean(x ** 2))

    if window_size < 1:
        window_size = 1

    rms_x = (
        df["flow_x_vel"]
        .rolling(window=window_size, center=True, min_periods=1)
        .apply(_rms, raw=True)
    )
    rms_y = (
        df["flow_y_vel"]
        .rolling(window=window_size, center=True, min_periods=1)
        .apply(_rms, raw=True)
    )
    df["flow_rms_mag"] = np.hypot(rms_x, rms_y)

    return df


# ---------------------------------------------------------------------------
# Adaptive threshold  (mirrors AdaptiveThreshold)
# ---------------------------------------------------------------------------

def compute_adaptive_threshold(
    df: pd.DataFrame,
    base_threshold: float,
    gain: float = 1.0,
    window_size: int = 50,
) -> pd.DataFrame:
    """Compute a per-sample adaptive threshold from optical-flow RMS.

    ``threshold_i = base_threshold + gain × flow_rms_mag_i``

    Requires ``flow_x_vel`` and ``flow_y_vel`` (calls
    ``compute_flow_window_rms`` internally if ``flow_rms_mag`` is not
    already present).

    Adds column: ``threshold``.

    Returns:
        The same DataFrame with added ``threshold`` column.
    """
    if "flow_rms_mag" not in df.columns:
        compute_flow_window_rms(df, window_size)

    df["threshold"] = base_threshold + gain * df["flow_rms_mag"]

    return df


# ---------------------------------------------------------------------------
# Relative dispersion  (mirrors RelativeDispersionCalculator)
# ---------------------------------------------------------------------------

def compute_relative_dispersion(
    df: pd.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
    window_size: int = 5,
    sample_duration_ms: float = 4.0,
) -> pd.DataFrame:
    """Compute gaze dispersion relative to optical-flow-integrated trajectory.

    For each temporal window the function integrates the optical-flow velocity
    forward and backward from the window centre to build an "ideal" gaze
    trajectory (where gaze *would* be if it perfectly tracked the background
    motion).  The dispersion of the *actual* gaze relative to this ideal
    trajectory is then computed as ``(max_rel_x − min_rel_x) + (max_rel_y −
    min_rel_y)``.

    Adds column: ``rel_dispersion``.

    Returns:
        The same DataFrame with added ``rel_dispersion`` column.
    """
    n = len(df)
    x_vals = df[x_col].values
    y_vals = df[y_col].values
    flow_x_vel = df["flow_x_vel"].values
    flow_y_vel = df["flow_y_vel"].values

    rel_dispersion = np.zeros(n)

    half_w = window_size // 2

    for i in range(n):
        start = max(0, i - half_w)
        end = min(n, i + half_w + 1)

        center_local = i - start  # index within the window slice

        wx = x_vals[start:end]
        wy = y_vals[start:end]
        wfx = flow_x_vel[start:end]
        wfy = flow_y_vel[start:end]

        center_x = wx[center_local]
        center_y = wy[center_local]

        dx = wfx * sample_duration_ms / 1000.0
        dy = wfy * sample_duration_ms / 1000.0

        wlen = len(wx)
        traj_x = np.zeros(wlen)
        traj_y = np.zeros(wlen)

        # Forward: integrate from center+1 to end
        if center_local + 1 < wlen:
            traj_x[center_local + 1:] = np.cumsum(dx[center_local + 1:])
            traj_y[center_local + 1:] = np.cumsum(dy[center_local + 1:])

        # Backward: integrate from center-1 to start
        if center_local > 0:
            traj_x[:center_local] = -np.cumsum(dx[:center_local][::-1])[::-1]
            traj_y[:center_local] = -np.cumsum(dy[:center_local][::-1])[::-1]

        ideal_x = center_x + traj_x
        ideal_y = center_y + traj_y

        rel_x = wx - ideal_x
        rel_y = wy - ideal_y

        rel_dispersion[i] = (np.nanmax(rel_x) - np.nanmin(rel_x)) + (np.nanmax(rel_y) - np.nanmin(rel_y))

    df["rel_dispersion"] = rel_dispersion

    return df


# ---------------------------------------------------------------------------
# Standard rolling dispersion  (mirrors DispersionCalculator)
# ---------------------------------------------------------------------------

def compute_dispersion(
    df: pd.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
    window_size: int = 5,
) -> pd.DataFrame:
    """Compute rolling spatial dispersion of gaze coordinates.

    Dispersion = (max_x − min_x) + (max_y − min_y) over a centred window.

    Adds column: ``dispersion``.

    Returns:
        The same DataFrame with added ``dispersion`` column.
    """
    x_spread = (
        df[x_col]
        .rolling(window_size, center=True, min_periods=1)
        .apply(lambda v: v.max() - v.min(), raw=True)
    )
    y_spread = (
        df[y_col]
        .rolling(window_size, center=True, min_periods=1)
        .apply(lambda v: v.max() - v.min(), raw=True)
    )
    df["dispersion"] = x_spread + y_spread

    return df


# ---------------------------------------------------------------------------
# Full preprocessing pipeline  (convenience entry-point)
# ---------------------------------------------------------------------------

def preprocess_gaze_data(
    df: pd.DataFrame,
    sampling_rate: float,
    *,
    is_velocity_based: bool = True,
    base_threshold: float = 150.0,
    adapt: bool = False,
    gain: float = 1.0,
    window_size_ms: float = 500.0,
    savgol_window_ms: float = 55.0,
) -> dict:
    """Run the full I-VAT+Frel preprocessing pipeline on *df* (in‑place).

    Steps performed (matching the research ``algorithms/`` code):

    1. Savitzky-Golay smoothing → ``filter_x``, ``filter_y``
    2. Gaze velocity on smoothed coords → ``vel_mag``
    3. If optical-flow columns present:
       a. Flow velocity → ``flow_x_vel``, ``flow_y_vel``
       b. Relative velocity → ``vel_rel_mag``
       c. (Optional) Adaptive threshold → per-sample ``threshold``
    4. If velocity-based=False (I-DT):
       a. Dispersion (on smoothed coords)
       b. Relative dispersion if flow is available

    Returns a metadata dict:
        - ``has_flow``: bool — whether optical-flow compensation was applied
        - ``vel_col``: str — velocity column name to threshold against
          (``"vel_rel_mag"`` or ``"vel_mag"``)
        - ``disp_col``: str — dispersion column name to threshold against
          (``"rel_dispersion"`` or ``"dispersion"``)
        - ``has_adaptive_threshold``: bool — whether a per-sample
          ``threshold`` column exists
    """
    has_flow = detect_flow_columns(df)

    # 1. Savgol filter
    apply_savgol_filter(df, sampling_rate, window_size_ms=savgol_window_ms)

    # Use smoothed coordinates for downstream calculations
    coord_x = "filter_x"
    coord_y = "filter_y"

    # 2. Gaze velocity (using average time delta, matching research code)
    compute_gaze_velocity(df, x_col=coord_x, y_col=coord_y)

    # 3. Flow compensation (if available)
    vel_col = "vel_mag"
    disp_col = "dispersion"
    has_adaptive_threshold = False

    if has_flow:
        compute_flow_velocity(df)
        compute_relative_velocity(df)
        vel_col = "vel_rel_mag"

        if adapt:
            sample_duration_ms = 1000.0 / sampling_rate
            window_size_samples = max(1, int(window_size_ms // sample_duration_ms))
            compute_adaptive_threshold(
                df,
                base_threshold=base_threshold,
                gain=gain,
                window_size=window_size_samples,
            )
            has_adaptive_threshold = True

    # 4. Dispersion (for I-DT)
    if not is_velocity_based:
        sample_duration_ms = 1000.0 / sampling_rate
        window_size_samples = max(3, int(25.0 // sample_duration_ms))

        compute_dispersion(df, x_col=coord_x, y_col=coord_y, window_size=window_size_samples)

        if has_flow:
            compute_relative_dispersion(
                df,
                x_col=coord_x,
                y_col=coord_y,
                window_size=window_size_samples,
                sample_duration_ms=sample_duration_ms,
            )
            disp_col = "rel_dispersion"

            if adapt and not has_adaptive_threshold:
                ws = max(1, int(window_size_ms // sample_duration_ms))
                compute_adaptive_threshold(
                    df,
                    base_threshold=base_threshold,
                    gain=gain,
                    window_size=ws,
                )
                has_adaptive_threshold = True

    return {
        "has_flow": has_flow,
        "vel_col": vel_col,
        "disp_col": disp_col,
        "has_adaptive_threshold": has_adaptive_threshold,
    }
