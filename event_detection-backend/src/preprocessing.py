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
# Temporal segment detection
# ---------------------------------------------------------------------------

def find_temporal_segments(
    df: pd.DataFrame,
    sampling_rate: float,
    gap_factor: float = 3.0,
    time_col: str = "timestamp",
) -> list:
    """Identify contiguous temporal segments separated by large gaps.

    After NaN rows have been removed, the remaining data may contain
    temporal discontinuities (e.g. from blinks or tracking loss).  This
    function detects those gaps and returns a list of ``(start, end)``
    index pairs (using *positional* indices into ``df``, i.e.
    ``range(len(df))``) that partition the data into segments of
    approximately uniform sampling.

    A gap is declared whenever the time difference between two
    consecutive rows exceeds ``gap_factor × expected_dt``.

    Args:
        df: DataFrame with a ``timestamp`` column (in ms).
        sampling_rate: Recording sampling rate in Hz.
        gap_factor: Multiplier on the expected frame duration to
            determine the minimum gap size.  The default (3.0) means a
            gap is flagged when Δt > 3× the expected inter-sample
            interval — this tolerates small jitter while catching real
            tracking loss.
        time_col: Name of the timestamp column.

    Returns:
        A list of ``(start, end)`` tuples where *start* is inclusive
        and *end* is exclusive (suitable for slicing::

            for s, e in segments:
                chunk = array[s:e]
        ).
    """
    expected_dt = 1000.0 / sampling_rate
    gap_threshold = gap_factor * expected_dt

    timestamps = df[time_col].values
    n = len(timestamps)
    if n == 0:
        return []

    dt = np.diff(timestamps)
    gap_indices = np.where(dt > gap_threshold)[0]  # indices where gap follows

    segments = []
    seg_start = 0
    for gi in gap_indices:
        segments.append((seg_start, gi + 1))  # up to and including gi
        seg_start = gi + 1
    segments.append((seg_start, n))  # last segment

    return segments


# ---------------------------------------------------------------------------
# Savitzky-Golay smoothing  (mirrors algorithms/transformations/savgol_filter.py)
# ---------------------------------------------------------------------------

def apply_savgol_filter(
    df: pd.DataFrame,
    sampling_rate: float,
    window_size_ms: float = 55,
    polyorder: int = 3,
    segments: list = None,
) -> pd.DataFrame:
    """Apply a Savitzky-Golay low-pass filter to gaze coordinates.

    The filter is applied **independently to each temporal segment** so
    that data from opposite sides of a blink or tracking-loss gap is
    never blended together.  If *segments* is ``None`` the function
    falls back to computing segments itself (or treating the whole
    array as one segment when no *sampling_rate* is provided).

    Adds ``filter_x`` and ``filter_y`` columns to *df* (in-place).

    Args:
        df: DataFrame with ``x`` and ``y`` columns.
        sampling_rate: Recording sampling rate in Hz.
        window_size_ms: Filter window size in milliseconds.
        polyorder: Polynomial order for the Savitzky-Golay filter.
        segments: Pre-computed temporal segments from
            :func:`find_temporal_segments`.  When ``None`` the segments
            are computed internally.

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

    if segments is None:
        segments = find_temporal_segments(df, sampling_rate)

    x_raw = df["x"].values
    y_raw = df["y"].values
    filter_x = x_raw.copy()
    filter_y = y_raw.copy()

    for seg_start, seg_end in segments:
        seg_len = seg_end - seg_start
        x_seg = x_raw[seg_start:seg_end]
        y_seg = y_raw[seg_start:seg_end]

        # Replace any remaining NaN within the segment for filtering
        x_seg = np.where(np.isnan(x_seg), 0, x_seg)
        y_seg = np.where(np.isnan(y_seg), 0, y_seg)

        # Determine the usable window for this segment
        seg_win = window_size
        if seg_len < polyorder + 2:
            # Too short to filter — keep raw values
            filter_x[seg_start:seg_end] = x_seg
            filter_y[seg_start:seg_end] = y_seg
            continue
        if seg_win > seg_len:
            seg_win = seg_len if seg_len % 2 == 1 else seg_len - 1
        if seg_win < polyorder + 2:
            filter_x[seg_start:seg_end] = x_seg
            filter_y[seg_start:seg_end] = y_seg
            continue

        filter_x[seg_start:seg_end] = savgol_filter(
            x_seg, window_length=seg_win, polyorder=polyorder
        )
        filter_y[seg_start:seg_end] = savgol_filter(
            y_seg, window_length=seg_win, polyorder=polyorder
        )

    df["filter_x"] = filter_x
    df["filter_y"] = filter_y

    return df


# ---------------------------------------------------------------------------
# Gaze velocity  (mirrors algorithms/transformations/velocity_calculator.py)
# ---------------------------------------------------------------------------

def compute_gaze_velocity(
    df: pd.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
    time_col: str = "timestamp",
    segments: list = None,
    sampling_rate: float = None,
) -> pd.DataFrame:
    """Compute gaze velocity using the *average* time delta.

    This mirrors the research code's ``VelocityCalculator`` which divides
    spatial deltas by the mean Δt across the recording (rather than per-sample
    Δt).  Using a constant denominator reduces noise in the velocity signal
    when the sampling interval varies slightly.

    When *segments* are provided, the velocity at the **last sample of each
    segment** is set to ``NaN`` so that the diff never spans a temporal gap
    (e.g. a blink).  Downstream code is expected to handle these NaN values
    (typically via ``.fillna(0)``).

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

    x_vel = (x_delta / avg_delta).values
    y_vel = (y_delta / avg_delta).values

    # Null out velocities at segment boundaries (the diff spans a gap)
    if segments is None and sampling_rate is not None:
        segments = find_temporal_segments(df, sampling_rate)
    if segments is not None:
        for seg_start, seg_end in segments:
            # Last sample of a segment: diff(-1) looks at the next sample
            # which belongs to a different segment (or doesn't exist).
            if seg_end - 1 >= 0 and seg_end < len(df):
                x_vel[seg_end - 1] = np.nan
                y_vel[seg_end - 1] = np.nan
            # First sample of a segment (except the very first): the
            # preceding sample was the last of the previous segment and
            # already NaN, but mark this side too for symmetry in
            # downstream rolling computations.
            if seg_start > 0:
                x_vel[seg_start] = np.nan
                y_vel[seg_start] = np.nan

    df["x_vel"] = x_vel
    df["y_vel"] = y_vel
    df["vel_mag"] = np.hypot(df["x_vel"], df["y_vel"])

    return df


# ---------------------------------------------------------------------------
# Optical-flow velocity  (mirrors algorithms/transformations/utils.py  add_flow_velocity)
# ---------------------------------------------------------------------------

def compute_flow_velocity(
    df: pd.DataFrame,
    segments: list = None,
    sampling_rate: float = None,
) -> pd.DataFrame:
    """Convert per-frame optical-flow displacements to velocities.

    Expects ``flow_x``, ``flow_y`` columns (displacement in pixels) and a
    time reference column.  If ``video_timestamp`` exists it is used; otherwise
    the regular ``timestamp`` column provides the time deltas.

    Flow velocities at temporal-gap boundaries are set to ``NaN`` (the
    flow displacement across a blink is meaningless).

    Adds columns: ``flow_x_vel``, ``flow_y_vel``, ``flow_vel_mag``.

    Returns:
        The same DataFrame with added flow velocity columns.
    """
    time_col = "video_timestamp" if "video_timestamp" in df.columns else "timestamp"

    flow_t_delta = df[time_col].diff(1)
    flow_t_delta = flow_t_delta.replace(0, np.nan).ffill().bfill().fillna(1)

    fx = (df["flow_x"] / flow_t_delta).values
    fy = (df["flow_y"] / flow_t_delta).values

    # Replace inf values from potential zero-division
    fx = np.where(np.isinf(fx), 0, fx)
    fy = np.where(np.isinf(fy), 0, fy)

    # Null out gap boundaries
    if segments is None and sampling_rate is not None:
        segments = find_temporal_segments(df, sampling_rate)
    if segments is not None:
        for seg_start, _seg_end in segments:
            if seg_start > 0:  # first sample after a gap
                fx[seg_start] = np.nan
                fy[seg_start] = np.nan

    df["flow_x_vel"] = fx
    df["flow_y_vel"] = fy
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

def compute_flow_window_rms(
    df: pd.DataFrame,
    window_size: int,
    segments: list = None,
    sampling_rate: float = None,
) -> pd.DataFrame:
    """Compute the rolling RMS of optical-flow velocity.

    The RMS magnitude is used as an indicator of how much the background is
    moving in a local temporal neighbourhood.  When *segments* are provided
    the rolling window is applied **per segment** so it never spans a
    temporal gap.

    Adds column: ``flow_rms_mag``.

    Returns:
        The same DataFrame with added ``flow_rms_mag`` column.
    """

    def _rms(x: np.ndarray) -> float:
        return np.sqrt(np.mean(x ** 2))

    if window_size < 1:
        window_size = 1

    if segments is None and sampling_rate is not None:
        segments = find_temporal_segments(df, sampling_rate)

    if segments is not None and len(segments) > 1:
        rms_mag = np.zeros(len(df))
        fx_vals = df["flow_x_vel"].values
        fy_vals = df["flow_y_vel"].values
        for seg_start, seg_end in segments:
            seg_fx = pd.Series(fx_vals[seg_start:seg_end])
            seg_fy = pd.Series(fy_vals[seg_start:seg_end])
            rx = seg_fx.rolling(window=min(window_size, seg_end - seg_start),
                                center=True, min_periods=1).apply(_rms, raw=True)
            ry = seg_fy.rolling(window=min(window_size, seg_end - seg_start),
                                center=True, min_periods=1).apply(_rms, raw=True)
            rms_mag[seg_start:seg_end] = np.hypot(rx.values, ry.values)
        df["flow_rms_mag"] = rms_mag
    else:
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
    segments: list = None,
    sampling_rate: float = None,
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
        compute_flow_window_rms(df, window_size, segments=segments,
                                sampling_rate=sampling_rate)

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
    segments: list = None,
    sampling_rate: float = None,
) -> pd.DataFrame:
    """Compute gaze dispersion relative to optical-flow-integrated trajectory.

    For each temporal window the function integrates the optical-flow velocity
    forward and backward from the window centre to build an "ideal" gaze
    trajectory (where gaze *would* be if it perfectly tracked the background
    motion).  The dispersion of the *actual* gaze relative to this ideal
    trajectory is then computed as ``(max_rel_x − min_rel_x) + (max_rel_y −
    min_rel_y)``.

    When *segments* are provided, each sample's window is clamped to its
    own segment so it never reaches across a temporal gap.

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

    # Build a per-sample segment-boundary lookup for fast clamping
    if segments is None and sampling_rate is not None:
        segments = find_temporal_segments(df, sampling_rate)

    seg_starts = np.zeros(n, dtype=int)
    seg_ends = np.full(n, n, dtype=int)
    if segments is not None:
        for s_start, s_end in segments:
            seg_starts[s_start:s_end] = s_start
            seg_ends[s_start:s_end] = s_end

    for i in range(n):
        # Clamp window to segment boundaries
        s_lo = seg_starts[i]
        s_hi = seg_ends[i]
        start = max(s_lo, i - half_w)
        end = min(s_hi, i + half_w + 1)

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
    segments: list = None,
    sampling_rate: float = None,
) -> pd.DataFrame:
    """Compute rolling spatial dispersion of gaze coordinates.

    Dispersion = (max_x − min_x) + (max_y − min_y) over a centred window.
    When *segments* are provided the rolling window is applied
    per-segment so it never spans a temporal gap.

    Adds column: ``dispersion``.

    Returns:
        The same DataFrame with added ``dispersion`` column.
    """
    if segments is None and sampling_rate is not None:
        segments = find_temporal_segments(df, sampling_rate)

    if segments is not None and len(segments) > 1:
        disp = np.zeros(len(df))
        x_vals = df[x_col].values
        y_vals = df[y_col].values
        for seg_start, seg_end in segments:
            seg_len = seg_end - seg_start
            sx = pd.Series(x_vals[seg_start:seg_end])
            sy = pd.Series(y_vals[seg_start:seg_end])
            win = min(window_size, seg_len)
            xs = sx.rolling(win, center=True, min_periods=1).apply(
                lambda v: v.max() - v.min(), raw=True
            )
            ys = sy.rolling(win, center=True, min_periods=1).apply(
                lambda v: v.max() - v.min(), raw=True
            )
            disp[seg_start:seg_end] = xs.values + ys.values
        df["dispersion"] = disp
    else:
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

    # 0. Detect temporal segments once — shared by all downstream steps
    segments = find_temporal_segments(df, sampling_rate)

    # 1. Savgol filter
    apply_savgol_filter(df, sampling_rate, window_size_ms=savgol_window_ms,
                        segments=segments)

    # Use smoothed coordinates for downstream calculations
    coord_x = "filter_x"
    coord_y = "filter_y"

    # 2. Gaze velocity (using average time delta, matching research code)
    compute_gaze_velocity(df, x_col=coord_x, y_col=coord_y,
                          segments=segments)

    # 3. Flow compensation (if available)
    vel_col = "vel_mag"
    disp_col = "dispersion"
    has_adaptive_threshold = False

    if has_flow:
        compute_flow_velocity(df, segments=segments)
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
                segments=segments,
            )
            has_adaptive_threshold = True

    # 4. Dispersion (for I-DT)
    if not is_velocity_based:
        sample_duration_ms = 1000.0 / sampling_rate
        window_size_samples = max(3, int(25.0 // sample_duration_ms))

        compute_dispersion(df, x_col=coord_x, y_col=coord_y,
                           window_size=window_size_samples, segments=segments)

        if has_flow:
            compute_relative_dispersion(
                df,
                x_col=coord_x,
                y_col=coord_y,
                window_size=window_size_samples,
                sample_duration_ms=sample_duration_ms,
                segments=segments,
            )
            disp_col = "rel_dispersion"

            if adapt and not has_adaptive_threshold:
                ws = max(1, int(window_size_ms // sample_duration_ms))
                compute_adaptive_threshold(
                    df,
                    base_threshold=base_threshold,
                    gain=gain,
                    window_size=ws,
                    segments=segments,
                )
                has_adaptive_threshold = True

    return {
        "has_flow": has_flow,
        "vel_col": vel_col,
        "disp_col": disp_col,
        "has_adaptive_threshold": has_adaptive_threshold,
    }
