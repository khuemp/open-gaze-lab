"""Fixation/saccade classifiers: I-DT (dispersion threshold) and I-VT
(velocity threshold).

Both algorithms share the preparation step in
:func:`prepare_classification_data`, which optionally runs the I-VAT+Frel
preprocessing pipeline (Savgol smoothing, gaze/flow velocity, relative
dispersion, adaptive threshold) when a sampling rate and optical-flow
columns are available. Without those inputs the classifiers fall back to
their legacy point-to-point velocity / expanding-window dispersion paths.
"""

import numpy as np
import pandas as pd

from .feature_extraction import preprocess_gaze_data
from .utils import compute_velocity, compute_mad


# ---------------------------------------------------------------------------
# Shared preparation / finalization
# ---------------------------------------------------------------------------

def prepare_classification_data(gaze_data: pd.DataFrame,
                                threshold: float,
                                adapt: bool = False,
                                tuning_parameter: float = 0.1,
                                is_velocity_based: bool = True,
                                sampling_rate: float = None) -> tuple:
    """Build the shared inputs for ``classify_idt`` / ``classify_ivt``.

    With *sampling_rate* and optical-flow columns the I-VAT+Frel pipeline
    runs in :func:`preprocess_gaze_data`. Otherwise only output buffers are
    allocated, optionally with a MAD-based threshold adaptation.

    Returns ``(result_data, threshold, preprocess_meta, arrays)``.
    """
    result_data = gaze_data.copy()
    n = len(result_data)
    has_flow = "flow_x" in result_data.columns and "flow_y" in result_data.columns

    preprocess_meta = None
    if sampling_rate is not None and sampling_rate > 0 and has_flow:
        preprocess_meta = preprocess_gaze_data(
            result_data,
            sampling_rate,
            is_velocity_based=is_velocity_based,
            base_threshold=threshold,
            adapt=adapt,
            gain=1.0,
            window_size_ms=500.0,
            savgol_window_ms=55.0,
        )

    # MAD fallback only when no per-sample adaptive threshold was produced.
    if adapt and (preprocess_meta is None or not preprocess_meta.get("has_adaptive_threshold", False)):
        velocity = compute_velocity(pd.DataFrame({
            "x": result_data["x"].values,
            "y": result_data["y"].values,
            "timestamp": result_data["timestamp"].values,
        }))
        if len(velocity) > 0:
            mad_velocity = compute_mad(velocity)
            if mad_velocity > 0:
                threshold = threshold * (1 + tuning_parameter * mad_velocity)

    arrays = {
        "event_type":     np.full(n, "Saccade", dtype="U10"),
        "fixation_x":     np.full(n, np.nan),
        "fixation_y":     np.full(n, np.nan),
        "event_duration": np.full(n, np.nan),
        "fixation_ids":   np.full(n, np.nan),
        "saccade_ids":    np.full(n, np.nan),
    }
    return result_data, threshold, preprocess_meta, arrays


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

def classify_idt(gaze_data, dispersion_threshold=150.0, min_fixation_duration=50,
                 adapt=False, tuning_parameter=0.1, sampling_rate=None):
    """I-DT (dispersion threshold) fixation/saccade classifier.

    With ``sampling_rate`` and optical-flow columns the I-VAT+Frel pipeline
    runs and per-sample dispersion is read from ``rel_dispersion``;
    otherwise the legacy expanding-window dispersion on raw coordinates is used.
    """
    result_data, dispersion_threshold, preprocess_meta, arrays = prepare_classification_data(
        gaze_data, dispersion_threshold, adapt, tuning_parameter,
        is_velocity_based=False, sampling_rate=sampling_rate,
    )

    n = len(result_data)
    x = result_data["x"].values
    y = result_data["y"].values
    t = result_data["timestamp"].values
    cx, cy = _select_centroid_coords(result_data, x, y)

    if preprocess_meta is not None:
        disp_values = result_data[preprocess_meta["disp_col"]].fillna(0).values
        has_adaptive = preprocess_meta["has_adaptive_threshold"]
        adaptive_thresh = result_data["threshold"].values if has_adaptive else None
    else:
        disp_values = None
        has_adaptive = False
        adaptive_thresh = None

    start_idx = 0
    fixation_id = 1

    while start_idx < n:
        if disp_values is not None:
            # Enhanced path: per-sample dispersion already computed.
            local_thresh = adaptive_thresh[start_idx] if has_adaptive else dispersion_threshold
            if disp_values[start_idx] > local_thresh:
                start_idx += 1
                continue

            current_idx = start_idx
            while current_idx < n:
                local_thresh = adaptive_thresh[current_idx] if has_adaptive else dispersion_threshold
                if disp_values[current_idx] > local_thresh:
                    break
                current_idx += 1
            end_idx = current_idx - 1
        else:
            # Legacy expanding-window dispersion on raw coordinates.
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

def classify_ivt(gaze_data, velocity_threshold=0.3, min_fixation_duration=50,
                 adapt=False, tuning_parameter=0.1, sampling_rate=None):
    """I-VT (velocity threshold) fixation/saccade classifier.

    With ``sampling_rate`` and optical-flow columns the I-VAT+Frel pipeline
    runs and the per-sample velocity column ``vel_rel_mag`` (or ``vel_mag``
    without flow) is used; otherwise the legacy point-to-point velocity is used.
    """
    result_data, velocity_threshold, preprocess_meta, arrays = prepare_classification_data(
        gaze_data, velocity_threshold, adapt, tuning_parameter,
        is_velocity_based=True, sampling_rate=sampling_rate,
    )

    n = len(result_data)
    x = result_data["x"].values
    y = result_data["y"].values
    t = result_data["timestamp"].values
    cx, cy = _select_centroid_coords(result_data, x, y)

    if preprocess_meta is not None:
        velocity = result_data[preprocess_meta["vel_col"]].fillna(0).values
        has_adaptive = preprocess_meta["has_adaptive_threshold"]
        adaptive_thresh = result_data["threshold"].values if has_adaptive else None
    else:
        velocity = compute_velocity(pd.DataFrame({"x": x, "y": y, "timestamp": t}))
        has_adaptive = False
        adaptive_thresh = None

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
