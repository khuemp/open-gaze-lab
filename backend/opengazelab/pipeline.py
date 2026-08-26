"""End-to-end event-detection pipeline.

The :class:`EventDetection` class takes a raw gaze DataFrame, normalises
its column names / coordinate units / timestamps, runs a chosen detector
(I-DT or I-VT), and post-processes the result (fixation/saccade merge,
invalid-sample reinsertion, fixation cleaning).

AOI classification (``classify_aoi``) is exposed as a method for callers
that want to assign each fixation to an Area of Interest after detection.
"""

import logging

import numpy as np
import pandas as pd

from .algorithms import classify_idt, classify_ivt
from .utils import (
    clean_fixations,
    correct_timestamps,
    merge_fixations,
    merge_saccades,
    reinsert_invalid_points,
    separate_invalid_points,
)


_STANDARD_COLUMN_KEYS = ('x', 'y', 'timestamp', 'flow_x', 'flow_y', 'video_timestamp')


class EventDetection:
    """Detect fixations and saccades in a gaze recording.

    Args:
        loaded_gaze_df: DataFrame containing gaze data with at least x, y
            and timestamp columns (under whatever names *column_mapping*
            specifies).
        resolution: ``(width, height)`` of the recording / screen in pixels.
            Required — used both to denormalize coordinates (when needed)
            and to filter out-of-range samples.
        column_mapping: Optional ``{standard_name: actual_column_name}``
            mapping. Standard names are ``x``, ``y``, ``timestamp``,
            ``flow_x``, ``flow_y``, ``video_timestamp``. Unspecified keys
            assume the column is already named according to the standard.
        is_normalized: When ``True`` the x/y values are scaled by the
            resolution. ``None`` (default) auto-detects: any coordinate >2
            implies pixel data.
    """

    def __init__(self, loaded_gaze_df, resolution, column_mapping=None,
                 is_normalized=None):
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            level=logging.INFO,
        )

        self.gaze_data = loaded_gaze_df.copy()
        self.is_valid_data = True
        self.resolution = resolution

        self._apply_column_mapping(column_mapping or {})
        self._scale_to_resolution(is_normalized)
        self._normalize_timestamps()

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _apply_column_mapping(self, column_mapping):
        """Rename columns from caller-provided names to standard names."""
        rename_map = {}
        for std_name in _STANDARD_COLUMN_KEYS:
            actual = column_mapping.get(std_name)
            if actual and actual != std_name and actual in self.gaze_data.columns:
                rename_map[actual] = std_name
        if rename_map:
            self.gaze_data = self.gaze_data.rename(columns=rename_map)

    def _scale_to_resolution(self, is_normalized):
        """Multiply normalized x/y by resolution (auto-detects when ``None``)."""
        if is_normalized is None:
            x_values = self.gaze_data['x'].dropna()
            y_values = self.gaze_data['y'].dropna()
            if len(x_values) > 0 and len(y_values) > 0:
                is_normalized = not (x_values.max() > 2 or y_values.max() > 2)
            else:
                is_normalized = True

        if is_normalized:
            self.gaze_data['x'] *= self.resolution[0]
            self.gaze_data['y'] *= self.resolution[1]

    def _normalize_timestamps(self):
        """Convert ``timestamp`` to milliseconds (heuristic on median delta-t).

        * median delta-t > 100 -> assume epoch seconds: subtract first then x1000
        * median delta-t < 0.1 -> assume seconds-from-start: x1000
        * otherwise leave as-is (already milliseconds)

        Applies the same conversion to ``video_timestamp`` when present.
        """
        timestamps = self.gaze_data['timestamp']
        if len(timestamps) < 2:
            return

        median_interval = np.median(np.diff(timestamps.values[:100]))
        if median_interval > 100:
            ts0 = timestamps.iloc[0]
            self.gaze_data['timestamp'] = (timestamps - ts0) * 1000
            ts_unit = 'epoch'
        elif median_interval < 0.1:
            self.gaze_data['timestamp'] = timestamps * 1000
            ts_unit = 'seconds'
        else:
            ts_unit = 'milliseconds'

        if 'video_timestamp' in self.gaze_data.columns:
            vt = self.gaze_data['video_timestamp']
            if ts_unit == 'epoch':
                self.gaze_data['video_timestamp'] = (vt - vt.iloc[0]) * 1000
            elif ts_unit == 'seconds':
                self.gaze_data['video_timestamp'] = vt * 1000

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect_event(self, min_fixation_duration, algorithm, detection_threshold,
                     sampling_rate, fixation_merge_threshold=None,
                     *, adapt, gain, window_size_ms):
        """Run fixation/saccade detection on ``self.gaze_data``.

        All numerical parameters are required and must be supplied by the caller
        (typically validated at the API/form layer). Toggles (``fixation_merge_threshold``,
        ``adapt``) keep their off-defaults.

        Returns ``(events_df, threshold_used)`` or ``(None, None)`` on failure.
        """
        if not self.is_valid_data:
            return None, None

        valid_gaze, invalid_mask, invalid_reasons = separate_invalid_points(
            self.gaze_data, self.resolution
        )
        original_length = len(self.gaze_data)

        if len(valid_gaze) == 0:
            logging.warning("No valid gaze samples after filtering. Returning None.")
            return None, None

        try:
            if algorithm == "idt":
                data, threshold_used = classify_idt(
                    valid_gaze,
                    dispersion_threshold=detection_threshold,
                    min_fixation_duration=min_fixation_duration,
                    adapt=adapt,
                    sampling_rate=sampling_rate,
                    gain=gain,
                    window_size_ms=window_size_ms,
                )
            elif algorithm == "ivt":
                data, threshold_used = classify_ivt(
                    valid_gaze,
                    velocity_threshold=detection_threshold,
                    min_fixation_duration=min_fixation_duration,
                    adapt=adapt,
                    sampling_rate=sampling_rate,
                    gain=gain,
                    window_size_ms=window_size_ms,
                )
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")

            if fixation_merge_threshold is not None:
                data = merge_fixations(data, fixation_merge_threshold=fixation_merge_threshold)
            data = merge_saccades(data)

            data = reinsert_invalid_points(data, original_length, invalid_mask, invalid_reasons)

            # Restore original x/y/timestamp on invalid rows from source data.
            invalid_indices = invalid_mask[invalid_mask].index
            for col in ("x", "y", "timestamp"):
                if col in self.gaze_data.columns:
                    data.loc[invalid_indices, col] = self.gaze_data.loc[invalid_indices, col].values

        except Exception as e:
            logging.error(f"Error detecting events: {e}")
            return None, None

        return data, threshold_used

    def process_event(self, min_fixation_duration, algorithm, detection_threshold,
                      sampling_rate, fixation_merge_threshold: float = None,
                      *, adapt: bool, gain: float, window_size_ms: float,
                      correct_timestamps_flag: bool = True):
        """Run the full detection + cleaning pipeline.

        Stores results on ``self.event_data_df`` and ``self.best_threshold``.
        Returns the cleaned events DataFrame, or ``None`` on failure.

        Args:
            min_fixation_duration: Minimum fixation duration in ms (required).
            algorithm: ``"idt"`` or ``"ivt"`` (required).
            detection_threshold: Initial threshold — px for I-DT, px/ms for I-VT (required).
            sampling_rate: Recording sampling rate in Hz (required; pass ``None``
                explicitly only when calling for legacy data without preprocessing).
            fixation_merge_threshold: If set, merge fixations within this
                many pixels of each other.
            adapt: Enable adaptive threshold (per-sample with flow data,
                otherwise MAD-based fallback).
            correct_timestamps_flag: Apply uniform-interval timestamp
                correction (default ``True``). Set ``False`` for datasets
                with already-accurate timestamps (e.g. .npy from Pupil
                Invisible).
        """
        if sampling_rate is not None and correct_timestamps_flag:
            self.gaze_data = correct_timestamps(self.gaze_data, sampling_rate)

        event_gaze, threshold_used = self.detect_event(
            min_fixation_duration=min_fixation_duration,
            algorithm=algorithm,
            fixation_merge_threshold=fixation_merge_threshold,
            detection_threshold=detection_threshold,
            adapt=adapt,
            sampling_rate=sampling_rate,
            gain=gain,
            window_size_ms=window_size_ms,
        )

        if event_gaze is None:
            logging.error("Failed to process event data")
            self.event_data_df = None
            self.best_threshold = None
            self.threshold_range = None
            return None

        self.event_data_df = clean_fixations(event_gaze)
        self.best_threshold = threshold_used

        # Per-sample adaptive threshold (flow-RMS path) writes a `threshold`
        # column. Surface its range so callers can show "input + adapted range".
        self.threshold_range = None
        if "threshold" in self.event_data_df.columns:
            thr = self.event_data_df["threshold"].dropna()
            if len(thr) > 0:
                self.threshold_range = {
                    "min": float(thr.min()),
                    "max": float(thr.max()),
                    "mean": float(thr.mean()),
                }

        logging.info("Event data processing completed")
        return self.event_data_df

    # ------------------------------------------------------------------
    # AOI classification (post-processing)
    # ------------------------------------------------------------------

    def classify_aoi(self, gaze_data, aois, algorithm="weighted_bbox_attach"):
        """Assign each fixation to an Area of Interest.

        ``algorithm="standard"`` picks the first AOI rectangle that
        contains the fixation. ``algorithm="attach"`` picks the closest
        AOI by centroid distance.
        """
        fixations = gaze_data[["fixation_id", "fixation_x", "fixation_y"]].drop_duplicates()
        fixations = fixations[fixations["fixation_x"].notna() & fixations["fixation_y"].notna()]
        fixations["aoi_type"] = np.nan
        fixations["aoi"] = np.nan
        fixations["aoi_id"] = np.nan
        fixations.reset_index(drop=True, inplace=True)

        if algorithm == "standard":
            for index, row in fixations.iterrows():
                for aoi_index, aoi in aois.iterrows():
                    if (aoi["pos_x"] <= row["fixation_x"] <= aoi["pos_x"] + aoi["width"] and
                            aoi["pos_y"] <= row["fixation_y"] <= aoi["pos_y"] + aoi["height"]):
                        fixations.loc[index, "aoi_type"] = aoi["aoi_type"]
                        fixations.loc[index, "aoi"] = aoi["aoi"]
                        fixations.loc[index, "aoi_id"] = aoi_index
                        break
        elif algorithm == "attach":
            for index, row in fixations.iterrows():
                min_distance = float("inf")
                closest_aoi = None
                closest_aoi_index = None
                for aoi_index, aoi in aois.iterrows():
                    cx = aoi["pos_x"] + aoi["width"] / 2
                    cy = aoi["pos_y"] + aoi["height"] / 2
                    distance = np.hypot(row["fixation_x"] - cx, row["fixation_y"] - cy)
                    if distance < min_distance:
                        min_distance = distance
                        closest_aoi = aoi
                        closest_aoi_index = aoi_index
                if closest_aoi is not None:
                    fixations.loc[index, "aoi_type"] = closest_aoi["aoi_type"]
                    fixations.loc[index, "aoi"] = closest_aoi["aoi"]
                    fixations.loc[index, "aoi_id"] = closest_aoi_index
        else:
            raise ValueError(f"Unsupported AOI classification algorithm: {algorithm}")

        return pd.merge(
            gaze_data,
            fixations[["fixation_id", "aoi_type", "aoi", "aoi_id"]],
            on="fixation_id",
            how="left",
        )
