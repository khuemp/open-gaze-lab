"""Fixation/saccade classifiers: I-DT (dispersion threshold) and I-VT
(velocity threshold).

Both algorithms share the preparation step in
:func:`prepare_classification_data`, which optionally runs the head-mounted
preprocessing pipeline (Savgol smoothing, gaze/flow velocity, relative
dispersion, adaptive threshold) when a sampling rate and optical-flow
columns are available. Without those inputs the classifiers fall back to
their legacy point-to-point velocity / expanding-window dispersion paths.
"""

import numpy as np
import pandas as pd

from .feature_extraction import preprocess_gaze_data, apply_adaptive_threshold
from .utils import compute_velocity


# ---------------------------------------------------------------------------
# Shared preparation / finalization
# ---------------------------------------------------------------------------

def prepare_classification_data(gaze_data: pd.DataFrame,
                                threshold: float,
                                adapt: bool,
                                *,
                                use_ivt: bool,
                                sampling_rate: float,
                                gain: float,
                                window_size_ms: float) -> tuple:
    """Build the shared inputs for ``classify_idt`` / ``classify_ivt``.

    With *sampling_rate* and optical-flow columns the head-mounted pipeline
    runs in :func:`preprocess_gaze_data`. When *adapt* is set,
    :func:`apply_adaptive_threshold` then picks the flow-RMS per-sample path
    (writing a ``threshold`` column) or the MAD scalar fallback automatically.

    Returns ``(result_data, threshold, preprocess_flow_meta, has_adaptive_threshold, arrays)``.
    """
    result_data = gaze_data.copy()
    n = len(result_data)
    has_flow = "flow_x" in result_data.columns and "flow_y" in result_data.columns

    preprocess_flow_meta = None
    if has_flow:
        preprocess_flow_meta = preprocess_gaze_data(
            result_data,
            sampling_rate,
            use_ivt=use_ivt,
        )

    has_adaptive_threshold = False
    if adapt:
        threshold, has_adaptive_threshold = apply_adaptive_threshold(
            result_data,
            base_threshold=threshold,
            sampling_rate=sampling_rate,
            gain=gain,
            window_size_ms=window_size_ms,
        )

    arrays = {
        "event_type":     np.full(n, "Saccade", dtype="U10"),
        "fixation_x":     np.full(n, np.nan),
        "fixation_y":     np.full(n, np.nan),
        "event_duration": np.full(n, np.nan),
        "fixation_ids":   np.full(n, np.nan),
        "saccade_ids":    np.full(n, np.nan),
    }
    return result_data, threshold, preprocess_flow_meta, has_adaptive_threshold, arrays


def finalize_result_dataframe(result_data, arrays):
    """Attach the detection output arrays to *result_data* and return it."""
    result_data["event_type"]     = arrays["event_type"]
    result_data["fixation_x"]     = arrays["fixation_x"]
    result_data["fixation_y"]     = arrays["fixation_y"]
    result_data["event_duration"] = arrays["event_duration"]
    result_data["fixation_id"]    = arrays["fixation_ids"]
    result_data["saccade_id"]     = arrays["saccade_ids"]
    return result_data


def _add_saccade_ids(event_type, saccade_ids):
    """Assign sequential IDs to consecutive runs of Saccade samples."""
    saccade_id_counter = 1
    in_saccade = False
    for i in range(len(event_type)):
        if event_type[i] == "Saccade":
            in_saccade = True
            saccade_ids[i] = saccade_id_counter
        elif in_saccade:
            in_saccade = False
            saccade_id_counter += 1
    return saccade_ids


def _select_centroid_coords(result_data, x, y):
    """Use the smoothed coordinates for fixation centroids when available."""
    if "filter_x" in result_data.columns and "filter_y" in result_data.columns:
        return result_data["filter_x"].values, result_data["filter_y"].values
    return x, y


def _record_fixation(arrays, start_idx, end_idx, cx, cy, window_duration, fixation_id):
    """Tag samples ``[start_idx..end_idx]`` as a single Fixation event."""
    sl = slice(start_idx, end_idx + 1)
    arrays["event_type"][sl]     = "Fixation"
    arrays["fixation_x"][sl]     = float(np.mean(cx[start_idx:end_idx + 1]))
    arrays["fixation_y"][sl]     = float(np.mean(cy[start_idx:end_idx + 1]))
    arrays["event_duration"][sl] = window_duration
    arrays["fixation_ids"][sl]   = fixation_id


# ---------------------------------------------------------------------------
# I-DT (dispersion threshold)
# ---------------------------------------------------------------------------

def classify_idt(gaze_data, dispersion_threshold, min_fixation_duration, sampling_rate,
                 *, adapt, gain, window_size_ms):
    """I-DT (dispersion threshold) fixation/saccade classifier.

    With ``sampling_rate`` and optical-flow columns the head-mounted pipeline
    runs and per-sample dispersion is read from ``rel_dispersion``;
    otherwise the legacy expanding-window dispersion on raw coordinates is used.
    Pass ``sampling_rate=None`` to skip preprocessing entirely.
    """
    result_data, dispersion_threshold, preprocess_flow_meta, has_adaptive, arrays = prepare_classification_data(
        gaze_data, dispersion_threshold, adapt,
        use_ivt=False, sampling_rate=sampling_rate,
        gain=gain, window_size_ms=window_size_ms,
    )

    n = len(result_data)
    x = result_data["x"].values
    y = result_data["y"].values
    t = result_data["timestamp"].values
    cx, cy = _select_centroid_coords(result_data, x, y)

    disp_flow_values = (
        result_data[preprocess_flow_meta["disp_col"]].fillna(0).values
        if preprocess_flow_meta is not None else None
    )
    adaptive_thresh = result_data["threshold"].values if has_adaptive else None

    start_idx = 0
    fixation_id = 1

    while start_idx < n:
        if disp_flow_values is not None:
            # Enhanced path: per-sample dispersion already computed. (head-mounted)
            local_thresh = adaptive_thresh[start_idx] if has_adaptive else dispersion_threshold
            if disp_flow_values[start_idx] > local_thresh:
                start_idx += 1
                continue

            current_idx = start_idx
            while current_idx < n:
                local_thresh = adaptive_thresh[current_idx] if has_adaptive else dispersion_threshold
                if disp_flow_values[current_idx] > local_thresh:
                    break
                current_idx += 1
            end_idx = current_idx - 1
        else:
            # Legacy expanding-window dispersion on raw coordinates. (stationary)
            current_idx = start_idx
            min_x = max_x = x[start_idx]
            min_y = max_y = y[start_idx]
            while current_idx < n:
                cur_x = x[current_idx]
                cur_y = y[current_idx]
                if cur_x > max_x: max_x = cur_x
                if cur_x < min_x: min_x = cur_x
                if cur_y > max_y: max_y = cur_y
                if cur_y < min_y: min_y = cur_y
                if (max_x - min_x) + (max_y - min_y) > dispersion_threshold:
                    break
                current_idx += 1
            end_idx = current_idx - 1 if current_idx > start_idx else start_idx

        window_duration = t[end_idx] - t[start_idx] if end_idx > start_idx else 0.0
        if window_duration >= min_fixation_duration:
            _record_fixation(arrays, start_idx, end_idx, cx, cy, window_duration, fixation_id)
            fixation_id += 1

        start_idx = end_idx + 1 if current_idx > start_idx else start_idx + 1

    arrays["saccade_ids"] = _add_saccade_ids(arrays["event_type"], arrays["saccade_ids"])
    return finalize_result_dataframe(result_data, arrays), dispersion_threshold


# ---------------------------------------------------------------------------
# I-VT (velocity threshold)
# ---------------------------------------------------------------------------

def classify_ivt(gaze_data, velocity_threshold, min_fixation_duration, sampling_rate,
                 *, adapt, gain, window_size_ms):
    """I-VT (velocity threshold) fixation/saccade classifier.

    With ``sampling_rate`` and optical-flow columns the head-mounted pipeline
    runs and the per-sample velocity column ``vel_rel_mag`` (or ``vel_mag``
    without flow) is used; otherwise the legacy point-to-point velocity is used.
    Pass ``sampling_rate=None`` to skip preprocessing entirely.
    """
    result_data, velocity_threshold, preprocess_flow_meta, has_adaptive, arrays = prepare_classification_data(
        gaze_data, velocity_threshold, adapt,
        use_ivt=True, sampling_rate=sampling_rate,
        gain=gain, window_size_ms=window_size_ms,
    )

    n = len(result_data)
    x = result_data["x"].values
    y = result_data["y"].values
    t = result_data["timestamp"].values
    cx, cy = _select_centroid_coords(result_data, x, y)

    if preprocess_flow_meta is not None:
        velocity = result_data[preprocess_flow_meta["vel_col"]].fillna(0).values
    else:
        velocity = compute_velocity(pd.DataFrame({"x": x, "y": y, "timestamp": t}))
    adaptive_thresh = result_data["threshold"].values if has_adaptive else None

    start_idx = 0
    fixation_id = 1

    while start_idx < n:
        local_thresh = adaptive_thresh[start_idx] if has_adaptive else velocity_threshold
        if velocity[start_idx] > local_thresh:
            start_idx += 1
            continue

        current_idx = start_idx
        while current_idx < n:
            local_thresh = adaptive_thresh[current_idx] if has_adaptive else velocity_threshold
            if velocity[current_idx] > local_thresh:
                break
            current_idx += 1
        end_idx = current_idx - 1

        window_duration = t[end_idx] - t[start_idx] if end_idx > start_idx else 0.0
        if window_duration >= min_fixation_duration:
            _record_fixation(arrays, start_idx, end_idx, cx, cy, window_duration, fixation_id)
            fixation_id += 1

        start_idx = end_idx + 1

    arrays["saccade_ids"] = _add_saccade_ids(arrays["event_type"], arrays["saccade_ids"])
    return finalize_result_dataframe(result_data, arrays), velocity_threshold
