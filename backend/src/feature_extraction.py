"""I-VAT+Frel feature extraction for enhanced event detection.

Per-sample DSP transformations (Savitzky-Golay smoothing, gaze/flow
velocity, relative velocity, dispersion variants, adaptive threshold)
that the I-DT / I-VT classifiers consume when a sampling rate and
optical-flow columns are available.

Based on: *Strategies for enhancing automatic fixation detection in
head-mounted eye tracking* — the I-VAT+Frel variant (Adaptive Velocity
Threshold with Head-Motion Compensation).
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


# ---------------------------------------------------------------------------
# Column / segment helpers
# ---------------------------------------------------------------------------

def detect_flow_columns(df: pd.DataFrame) -> bool:
    """True when *df* has both ``flow_x`` and ``flow_y`` columns."""
    return "flow_x" in df.columns and "flow_y" in df.columns


def find_temporal_segments(
    df: pd.DataFrame,
    sampling_rate: float,
    gap_factor: float = 3.0,
    time_col: str = "timestamp",
) -> list:
    """Identify contiguous temporal segments separated by large gaps.

    Returns a list of ``(start, end)`` positional index pairs (``end`` is
    exclusive, suitable for slicing). A gap is declared whenever the time
    difference between consecutive rows exceeds ``gap_factor x expected_dt``
    (default tolerates 3x jitter while catching real tracking loss).
    """
    timestamps = df[time_col].values
    n = len(timestamps)
    if n == 0:
        return []

    gap_threshold = gap_factor * (1000.0 / sampling_rate)
    gap_indices = np.where(np.diff(timestamps) > gap_threshold)[0]

    segments = []
    seg_start = 0
    for gi in gap_indices:
        segments.append((seg_start, gi + 1))
        seg_start = gi + 1
    segments.append((seg_start, n))
    return segments


def _ensure_segments(df, segments, sampling_rate):
    """Compute segments from *sampling_rate* if not already provided."""
    if segments is None and sampling_rate is not None:
        return find_temporal_segments(df, sampling_rate)
    return segments


def _nan_segment_boundaries(values: np.ndarray, segments: list,
                            *, on_last: bool, on_first: bool) -> np.ndarray:
    """Set NaN at segment boundaries to prevent diffs spanning gaps.

    *on_last* nulls the last sample of every segment (where ``diff(-1)``
    would look at the next segment); *on_first* nulls the first sample of
    every segment except the very first.
    """
    n = len(values)
    for seg_start, seg_end in segments:
        if on_last and 0 <= seg_end - 1 and seg_end < n:
            values[seg_end - 1] = np.nan
        if on_first and seg_start > 0:
            values[seg_start] = np.nan
    return values


# ---------------------------------------------------------------------------
# Savitzky-Golay smoothing
# ---------------------------------------------------------------------------

def apply_savgol_filter(
    df: pd.DataFrame,
    sampling_rate: float,
    window_size_ms: float = 55,
    polyorder: int = 3,
    segments: list = None,
) -> pd.DataFrame:
    """Apply a Savitzky-Golay low-pass filter to gaze coordinates per segment.

    Adds ``filter_x`` and ``filter_y`` columns. The filter is applied
    independently to each temporal segment so blink-spanning data is never
    blended together.
    """
    frame_duration_ms = 1000.0 / sampling_rate
    window_size = int(window_size_ms // frame_duration_ms)
    if window_size < polyorder + 2:
        window_size = polyorder + 2
    if window_size % 2 == 0:
        window_size += 1

    segments = _ensure_segments(df, segments, sampling_rate)

    x_raw = df["x"].values
    y_raw = df["y"].values
    filter_x = x_raw.copy()
    filter_y = y_raw.copy()

    for seg_start, seg_end in segments:
        seg_len = seg_end - seg_start
        x_seg = np.where(np.isnan(x_raw[seg_start:seg_end]), 0, x_raw[seg_start:seg_end])
        y_seg = np.where(np.isnan(y_raw[seg_start:seg_end]), 0, y_raw[seg_start:seg_end])

        seg_win = window_size
        if seg_win > seg_len:
            seg_win = seg_len if seg_len % 2 == 1 else seg_len - 1

        if seg_len < polyorder + 2 or seg_win < polyorder + 2:
            filter_x[seg_start:seg_end] = x_seg
            filter_y[seg_start:seg_end] = y_seg
            continue

        filter_x[seg_start:seg_end] = savgol_filter(x_seg, window_length=seg_win, polyorder=polyorder)
        filter_y[seg_start:seg_end] = savgol_filter(y_seg, window_length=seg_win, polyorder=polyorder)

    df["filter_x"] = filter_x
    df["filter_y"] = filter_y
    return df


# ---------------------------------------------------------------------------
# Velocities
# ---------------------------------------------------------------------------

def compute_gaze_velocity(
    df: pd.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
    time_col: str = "timestamp",
    segments: list = None,
    sampling_rate: float = None,
) -> pd.DataFrame:
    """Gaze velocity with a constant denominator (mean delta-t over the recording).

    Mirrors the research code's ``VelocityCalculator``. Using mean delta-t
    rather than per-sample delta-t reduces noise from jitter. Velocities at
    segment boundaries are nulled so the diff never spans a temporal gap.

    Adds ``x_vel``, ``y_vel``, ``vel_mag``.
    """
    x_delta = -df[x_col].diff(-1).fillna(0)
    y_delta = -df[y_col].diff(-1).fillna(0)
    t_delta = -df[time_col].diff(-1).fillna(0)

    avg_delta = t_delta.mean()
    if avg_delta == 0 or np.isnan(avg_delta):
        avg_delta = 1.0  # safety

    x_vel = (x_delta / avg_delta).values
    y_vel = (y_delta / avg_delta).values

    segments = _ensure_segments(df, segments, sampling_rate)
    if segments is not None:
        _nan_segment_boundaries(x_vel, segments, on_last=True, on_first=True)
        _nan_segment_boundaries(y_vel, segments, on_last=True, on_first=True)

    df["x_vel"] = x_vel
    df["y_vel"] = y_vel
    df["vel_mag"] = np.hypot(x_vel, y_vel)
    return df


def compute_flow_velocity(
    df: pd.DataFrame,
    segments: list = None,
    sampling_rate: float = None,
) -> pd.DataFrame:
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

    segments = _ensure_segments(df, segments, sampling_rate)
    if segments is not None:
        _nan_segment_boundaries(fx, segments, on_last=False, on_first=True)
        _nan_segment_boundaries(fy, segments, on_last=False, on_first=True)

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


def compute_flow_window_rms(
    df: pd.DataFrame,
    window_size: int,
    segments: list = None,
    sampling_rate: float = None,
) -> pd.DataFrame:
    """Rolling RMS of optical-flow velocity (head-motion strength indicator).

    Per-segment when *segments* are provided so the window never spans a gap.
    Adds ``flow_rms_mag``.
    """
    window_size = max(1, window_size)
    segments = _ensure_segments(df, segments, sampling_rate)

    if segments is not None and len(segments) > 1:
        rms_mag = np.zeros(len(df))
        fx_vals = df["flow_x_vel"].values
        fy_vals = df["flow_y_vel"].values
        for seg_start, seg_end in segments:
            seg_len = seg_end - seg_start
            win = min(window_size, seg_len)
            rx = _rolling_rms(pd.Series(fx_vals[seg_start:seg_end]), win)
            ry = _rolling_rms(pd.Series(fy_vals[seg_start:seg_end]), win)
            rms_mag[seg_start:seg_end] = np.hypot(rx.values, ry.values)
        df["flow_rms_mag"] = rms_mag
    else:
        df["flow_rms_mag"] = np.hypot(
            _rolling_rms(df["flow_x_vel"], window_size),
            _rolling_rms(df["flow_y_vel"], window_size),
        )
    return df


def compute_adaptive_threshold(
    df: pd.DataFrame,
    base_threshold: float,
    gain: float = 1.0,
    window_size: int = 50,
    segments: list = None,
    sampling_rate: float = None,
) -> pd.DataFrame:
    """Per-sample threshold = ``base_threshold + gain x flow_rms_mag``.

    Computes ``flow_rms_mag`` first if missing. Adds ``threshold``.
    """
    if "flow_rms_mag" not in df.columns:
        compute_flow_window_rms(df, window_size, segments=segments,
                                sampling_rate=sampling_rate)
    df["threshold"] = base_threshold + gain * df["flow_rms_mag"]
    return df


# ---------------------------------------------------------------------------
# Dispersions
# ---------------------------------------------------------------------------

def compute_relative_dispersion(
    df: pd.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
    window_size: int = 5,
    sample_duration_ms: float = 4.0,
    segments: list = None,
    sampling_rate: float = None,
) -> pd.DataFrame:
    """Gaze dispersion relative to the optical-flow-integrated trajectory.

    For each sample's window the optical-flow velocity is integrated forward
    and backward from the centre to produce an "ideal" gaze trajectory (where
    gaze *would* be if it perfectly tracked background motion). The dispersion
    of actual gaze relative to this trajectory is
    ``(max_rel_x - min_rel_x) + (max_rel_y - min_rel_y)``. Windows are clamped
    to segment boundaries. Adds ``rel_dispersion``.
    """
    n = len(df)
    x_vals = df[x_col].values
    y_vals = df[y_col].values
    flow_x_vel = df["flow_x_vel"].values
    flow_y_vel = df["flow_y_vel"].values

    rel_dispersion = np.zeros(n)
    half_w = window_size // 2

    segments = _ensure_segments(df, segments, sampling_rate)
    seg_starts = np.zeros(n, dtype=int)
    seg_ends = np.full(n, n, dtype=int)
    if segments is not None:
        for s_start, s_end in segments:
            seg_starts[s_start:s_end] = s_start
            seg_ends[s_start:s_end] = s_end

    dt_s = sample_duration_ms / 1000.0

    for i in range(n):
        start = max(seg_starts[i], i - half_w)
        end = min(seg_ends[i], i + half_w + 1)
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


def compute_dispersion(
    df: pd.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
    window_size: int = 5,
    segments: list = None,
    sampling_rate: float = None,
) -> pd.DataFrame:
    """Rolling spatial dispersion ``(max_x - min_x) + (max_y - min_y)``.

    Per-segment when *segments* are provided. Adds ``dispersion``.
    """
    segments = _ensure_segments(df, segments, sampling_rate)
    spread = lambda v: v.max() - v.min()

    if segments is not None and len(segments) > 1:
        disp = np.zeros(len(df))
        x_vals = df[x_col].values
        y_vals = df[y_col].values
        for seg_start, seg_end in segments:
            seg_len = seg_end - seg_start
            win = min(window_size, seg_len)
            sx = pd.Series(x_vals[seg_start:seg_end])
            sy = pd.Series(y_vals[seg_start:seg_end])
            xs = sx.rolling(win, center=True, min_periods=1).apply(spread, raw=True)
            ys = sy.rolling(win, center=True, min_periods=1).apply(spread, raw=True)
            disp[seg_start:seg_end] = xs.values + ys.values
        df["dispersion"] = disp
    else:
        df["dispersion"] = (
            df[x_col].rolling(window_size, center=True, min_periods=1).apply(spread, raw=True)
            + df[y_col].rolling(window_size, center=True, min_periods=1).apply(spread, raw=True)
        )
    return df


# ---------------------------------------------------------------------------
# Pipeline entry-point
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
    """Run the I-VAT+Frel preprocessing pipeline on *df* (in-place).

    Only the columns the chosen classifier will actually consume are produced:

    * Always: Savitzky-Golay smoothing -> ``filter_x``/``filter_y``.
    * When flow is present: ``flow_x_vel``/``flow_y_vel`` (shared input
      for relative velocity, relative dispersion, and the adaptive RMS).
    * For I-VT: ``vel_mag`` (and ``vel_rel_mag`` with flow).
    * For I-DT: ``dispersion`` (no flow) **or** ``rel_dispersion`` (with flow).
    * If ``adapt`` and flow present: per-sample ``threshold`` column.

    Returns metadata pointing at the column names to threshold against
    (``vel_col`` / ``disp_col``) and whether ``threshold`` is per-sample.
    """
    has_flow = detect_flow_columns(df)
    segments = find_temporal_segments(df, sampling_rate)

    apply_savgol_filter(df, sampling_rate, window_size_ms=savgol_window_ms,
                        segments=segments)
    coord_x, coord_y = "filter_x", "filter_y"

    if has_flow:
        compute_flow_velocity(df, segments=segments)

    vel_col = "vel_mag"
    disp_col = "dispersion"

    if is_velocity_based:
        compute_gaze_velocity(df, x_col=coord_x, y_col=coord_y, segments=segments)
        if has_flow:
            compute_relative_velocity(df)
            vel_col = "vel_rel_mag"
    else:
        sample_duration_ms = 1000.0 / sampling_rate
        window_size_samples = max(3, int(25.0 // sample_duration_ms))
        if has_flow:
            compute_relative_dispersion(
                df, x_col=coord_x, y_col=coord_y,
                window_size=window_size_samples,
                sample_duration_ms=sample_duration_ms,
                segments=segments,
            )
            disp_col = "rel_dispersion"
        else:
            compute_dispersion(df, x_col=coord_x, y_col=coord_y,
                               window_size=window_size_samples, segments=segments)

    has_adaptive_threshold = False
    if adapt and has_flow:
        sample_duration_ms = 1000.0 / sampling_rate
        rms_window_samples = max(1, int(window_size_ms // sample_duration_ms))
        compute_adaptive_threshold(
            df, base_threshold=base_threshold, gain=gain,
            window_size=rms_window_samples, segments=segments,
        )
        has_adaptive_threshold = True

    return {
        "has_flow": has_flow,
        "vel_col": vel_col,
        "disp_col": disp_col,
        "has_adaptive_threshold": has_adaptive_threshold,
    }
