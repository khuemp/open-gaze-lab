"""Head-mounted feature extraction for enhanced event detection.

Per-sample DSP transformations (Savitzky-Golay smoothing, gaze/flow
velocity, relative velocity, dispersion variants, adaptive threshold)
that the I-DT / I-VT classifiers consume when a sampling rate and
optical-flow columns are available.

Based on: *Strategies for enhancing automatic fixation detection in
head-mounted eye tracking* — the head-mounted variant (Adaptive Velocity
Threshold with Head-Motion Compensation).
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from .utils import compute_velocity, compute_mad

# ---------------------------------------------------------------------------
# Savitzky-Golay smoothing
# ---------------------------------------------------------------------------

def apply_savgol_filter(
    df: pd.DataFrame,
    sampling_rate: float,
    window_size_ms: float = 55.0,
    polyorder: int = 3,
) -> pd.DataFrame:
    """Apply a Savitzky-Golay low-pass filter to gaze coordinates.

    Adds ``filter_x`` and ``filter_y`` columns.
    """
    frame_duration_ms = 1000.0 / sampling_rate
    window_size = int(window_size_ms // frame_duration_ms)
    if window_size < polyorder + 2:
        window_size = polyorder + 2
    if window_size % 2 == 0:
        window_size += 1

    x_raw = df["x"].values
    y_raw = df["y"].values
    n = len(x_raw)

    x_in = np.where(np.isnan(x_raw), 0, x_raw)
    y_in = np.where(np.isnan(y_raw), 0, y_raw)

    win = window_size
    if win > n:
        win = n if n % 2 == 1 else n - 1

    if n < polyorder + 2 or win < polyorder + 2:
        df["filter_x"] = x_in
        df["filter_y"] = y_in
    else:
        df["filter_x"] = savgol_filter(x_in, window_length=win, polyorder=polyorder)
        df["filter_y"] = savgol_filter(y_in, window_length=win, polyorder=polyorder)
    return df


# ---------------------------------------------------------------------------
# Velocities
# ---------------------------------------------------------------------------

def compute_gaze_velocity(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    time_col: str = "timestamp",
) -> pd.DataFrame:
    """Gaze velocity with a constant denominator (mean delta-t over the recording).

    Mirrors the research code's ``VelocityCalculator``. Using mean delta-t
    rather than per-sample delta-t reduces noise from jitter.

    Adds ``x_vel``, ``y_vel``, ``vel_mag``.
    """
    x_delta = -df[x_col].diff(-1).fillna(0)
    y_delta = -df[y_col].diff(-1).fillna(0)
    t_delta = -df[time_col].diff(-1).fillna(0)

    avg_delta = t_delta.mean()
    if avg_delta == 0 or np.isnan(avg_delta):
        avg_delta = 1.0

    x_vel = (x_delta / avg_delta).values
    y_vel = (y_delta / avg_delta).values

    df["x_vel"] = x_vel
    df["y_vel"] = y_vel
    df["vel_mag"] = np.hypot(x_vel, y_vel)
    return df


def compute_flow_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """Convert per-frame optical-flow displacements into velocities.

    Uses ``video_timestamp`` for delta-t when present (preferred — flow is
    sampled at the video frame rate), otherwise falls back to ``timestamp``.
    Adds ``flow_x_vel``, ``flow_y_vel``, ``flow_vel_mag``.
    """
    time_col = "video_timestamp" if "video_timestamp" in df.columns else "timestamp"
    flow_t_delta = df[time_col].diff(1).replace(0, np.nan).ffill().bfill().fillna(1)

    fx = (df["flow_x"] / flow_t_delta).values
    fy = (df["flow_y"] / flow_t_delta).values
    fx = np.where(np.isinf(fx), 0, fx)
    fy = np.where(np.isinf(fy), 0, fy)

    df["flow_x_vel"] = fx
    df["flow_y_vel"] = fy
    df["flow_vel_mag"] = np.hypot(fx, fy)
    return df


def compute_relative_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """Subtract optical-flow velocity from gaze velocity (Frel compensation).

    Requires ``x_vel``/``y_vel`` and ``flow_x_vel``/``flow_y_vel``.
    Adds ``x_vel_rel``, ``y_vel_rel``, ``vel_rel_mag``.
    """
    df["x_vel_rel"] = df["x_vel"] - df["flow_x_vel"]
    df["y_vel_rel"] = df["y_vel"] - df["flow_y_vel"]
    df["vel_rel_mag"] = np.hypot(df["x_vel_rel"], df["y_vel_rel"])
    return df


# ---------------------------------------------------------------------------
# Adaptive threshold (flow-RMS based)
# ---------------------------------------------------------------------------

def _rolling_rms(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, center=True, min_periods=1).apply(
        lambda v: np.sqrt(np.mean(v ** 2)), raw=True
    )


def compute_flow_window_rms(df: pd.DataFrame, rms_window_size: int) -> pd.DataFrame:
    """Rolling RMS of optical-flow velocity (head-motion strength indicator).

    Adds ``flow_rms_mag``.
    """
    rms_window_size = max(1, rms_window_size)
    df["flow_rms_mag"] = np.hypot(
        _rolling_rms(df["flow_x_vel"], rms_window_size),
        _rolling_rms(df["flow_y_vel"], rms_window_size),
    )
    return df


def apply_adaptive_threshold(
    df: pd.DataFrame,
    base_threshold: float,
    *,
    sampling_rate: float,
    gain: float = 0.0,
    window_size_ms: float = 0.0,
    tuning_parameter: float = 0.1,
) -> tuple:
    """Adapt the saccade threshold to local motion, picking a strategy by signal.

    * **Flow-RMS, per-sample** when *df* already has ``flow_x_vel`` /
      ``flow_y_vel`` and a sampling rate is given. Writes a per-sample
      ``threshold`` column via :func:`compute_adaptive_threshold` and returns
      ``(base_threshold, True)`` — the scalar is unused; the classifier reads
      the per-sample column.
    * **MAD scalar fallback** otherwise. Scales *base_threshold* by the MAD
      of point-to-point gaze velocity and returns ``(adapted_threshold, False)``.
    """
    has_flow_vel = "flow_x_vel" in df.columns and "flow_y_vel" in df.columns

    if has_flow_vel and sampling_rate is not None and sampling_rate > 0:
        sample_duration_ms = 1000.0 / sampling_rate
        rms_window_size = int(window_size_ms / sample_duration_ms) # convert window size from ms to samples
        if "flow_rms_mag" not in df.columns:
            compute_flow_window_rms(df, rms_window_size)
        df["threshold"] = base_threshold + gain * df["flow_rms_mag"]
        return base_threshold, True

    velocity = compute_velocity(pd.DataFrame({
        "x": df["x"].values,
        "y": df["y"].values,
        "timestamp": df["timestamp"].values,
    }))
    if len(velocity) > 0:
        mad_velocity = compute_mad(velocity)
        if mad_velocity > 0:
            return base_threshold * (1 + tuning_parameter * mad_velocity), False
    return base_threshold, False


# ---------------------------------------------------------------------------
# Dispersions
# ---------------------------------------------------------------------------

def compute_relative_dispersion(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    window_size: int,
    sample_duration_ms: float,
) -> pd.DataFrame:
    """Gaze dispersion relative to the optical-flow-integrated trajectory.

    For each sample's window the optical-flow velocity is integrated forward
    and backward from the centre to produce an "ideal" gaze trajectory (where
    gaze *would* be if it perfectly tracked background motion). The dispersion
    of actual gaze relative to this trajectory is
    ``(max_rel_x - min_rel_x) + (max_rel_y - min_rel_y)``. Adds ``rel_dispersion``.
    """
    n = len(df)
    x_vals = df[x_col].values
    y_vals = df[y_col].values
    flow_x_vel = df["flow_x_vel"].values
    flow_y_vel = df["flow_y_vel"].values

    rel_dispersion = np.zeros(n)
    half_w = window_size // 2
    dt_s = sample_duration_ms / 1000.0

    for i in range(n):
        start = max(0, i - half_w)
        end = min(n, i + half_w + 1)
        center_local = i - start

        wx = x_vals[start:end]
        wy = y_vals[start:end]
        dx = flow_x_vel[start:end] * dt_s
        dy = flow_y_vel[start:end] * dt_s

        wlen = end - start
        traj_x = np.zeros(wlen)
        traj_y = np.zeros(wlen)

        if center_local + 1 < wlen:
            traj_x[center_local + 1:] = np.cumsum(dx[center_local + 1:])
            traj_y[center_local + 1:] = np.cumsum(dy[center_local + 1:])
        if center_local > 0:
            traj_x[:center_local] = -np.cumsum(dx[:center_local][::-1])[::-1]
            traj_y[:center_local] = -np.cumsum(dy[:center_local][::-1])[::-1]

        ideal_x = wx[center_local] + traj_x
        ideal_y = wy[center_local] + traj_y
        rel_x = wx - ideal_x
        rel_y = wy - ideal_y
        rel_dispersion[i] = (np.nanmax(rel_x) - np.nanmin(rel_x)) + (np.nanmax(rel_y) - np.nanmin(rel_y))

    df["rel_dispersion"] = rel_dispersion
    return df


# ---------------------------------------------------------------------------
# Pipeline entry-point
# ---------------------------------------------------------------------------

def preprocess_gaze_data(
    df: pd.DataFrame,
    sampling_rate: float,
    *,
    use_ivt: bool,
) -> dict:
    """Run the head-mounted feature pipeline on *df* (in-place).

    Only the columns the chosen classifier will consume are produced:

    * Always: Savitzky-Golay smoothing -> ``filter_x``/``filter_y``.
    * When flow is present: ``flow_x_vel``/``flow_y_vel`` (shared input
      for relative velocity, relative dispersion, and adaptive RMS).
    * For I-VT: ``vel_mag`` (and ``vel_rel_mag`` with flow).
    * For I-DT: ``dispersion`` (no flow) **or** ``rel_dispersion`` (with flow).

    The adaptive threshold is applied separately by
    :func:`apply_adaptive_threshold` after this pipeline runs.

    Returns metadata pointing at the column names to threshold against
    (``vel_col`` / ``disp_col``).
    """

    apply_savgol_filter(df, sampling_rate)
    coord_x, coord_y = "filter_x", "filter_y"

    compute_flow_velocity(df)

    vel_col = "vel_mag"
    disp_col = "dispersion"

    if use_ivt:
        compute_gaze_velocity(df, x_col=coord_x, y_col=coord_y)

        compute_relative_velocity(df)
        vel_col = "vel_rel_mag"
    else:
        sample_duration_ms = 1000.0 / sampling_rate
        dispersion_window_size_ms = 25.0
        window_size = int(dispersion_window_size_ms / sample_duration_ms) # convert window size from ms to samples
        compute_relative_dispersion(
            df, x_col=coord_x, y_col=coord_y,
            window_size=window_size,
            sample_duration_ms=sample_duration_ms,
        )
        disp_col = "rel_dispersion"

    return {
        "vel_col": vel_col,
        "disp_col": disp_col,
    }
