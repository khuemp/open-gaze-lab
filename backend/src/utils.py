"""Math and data-handling helpers used across the detection pipeline.

Grouped roughly into:

* Invalid-sample handling: ``separate_invalid_points``, ``reinsert_invalid_points``
* Timestamp / velocity primitives: ``correct_timestamps``, ``compute_velocity``,
  ``compute_mad``
* Post-processing of detection output: ``clean_fixations``, ``merge_fixations``,
  ``merge_saccades``
"""

import logging

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Invalid-sample handling
# ---------------------------------------------------------------------------

def separate_invalid_points(df, resolution):
    """Split *df* into valid samples plus a mask + reasons for invalid rows.

    A sample is invalid when its x or y is NaN, or when (x, y) lies outside
    the screen rectangle ``[0, res_w] x [0, res_h]``.

    Returns ``(valid_df, invalid_mask, reasons)`` where *valid_df* preserves
    the original index and *reasons* is ``'NaN'`` or
    ``'Out of Range Gaze Samples'`` per invalid row.
    """
    res_w, res_h = resolution

    is_nan = df['x'].isna() | df['y'].isna()
    is_oor = (
        (df['x'] < 0) | (df['x'] > res_w) |
        (df['y'] < 0) | (df['y'] > res_h)
    ) & ~is_nan
    invalid_mask = is_nan | is_oor

    reasons = pd.Series(np.nan, index=df.index, dtype=object)
    reasons[is_nan] = 'NaN'
    reasons[is_oor] = 'Out of Range Gaze Samples'

    n_nan = int(is_nan.sum())
    n_oor = int(is_oor.sum())
    if n_nan or n_oor:
        logging.info(
            f"Separated {n_nan} NaN and {n_oor} out-of-range gaze samples "
            f"from {len(df)} total rows"
        )

    return df[~invalid_mask].copy(), invalid_mask, reasons


def reinsert_invalid_points(valid_df, original_length, invalid_mask, reasons):
    """Restore invalid rows back into the processed DataFrame at original positions.

    Invalid rows get their reason as ``event_type`` and NaN for everything
    else. Numeric columns are coerced back to numeric dtype after the
    object-dtype merge.
    """
    full_df = pd.DataFrame(index=range(original_length), columns=valid_df.columns)

    valid_indices = invalid_mask[~invalid_mask].index
    full_df.loc[valid_indices] = valid_df.values

    invalid_indices = invalid_mask[invalid_mask].index
    full_df.loc[invalid_indices, 'event_type'] = reasons[invalid_indices].values

    for col in valid_df.columns:
        if col == 'event_type':
            continue
        full_df[col] = pd.to_numeric(full_df[col], errors='coerce')

    return full_df


# ---------------------------------------------------------------------------
# Timestamp / velocity primitives
# ---------------------------------------------------------------------------

def correct_timestamps(df, sampling_rate):
    """Regenerate timestamps at uniform ``1000/sampling_rate`` ms intervals.

    Useful when input timestamps are jittery; the first sample's timestamp
    is preserved as the reference point.
    """
    df_corrected = df.copy()
    interval_ms = 1000.0 / sampling_rate
    first_timestamp = df_corrected['timestamp'].iloc[0]
    df_corrected['timestamp'] = first_timestamp + np.arange(len(df_corrected)) * interval_ms
    return df_corrected


def compute_velocity(df):
    """Point-to-point Euclidean velocity (px/ms).

    Handles variable framerates without modifying timestamps. NaN coordinates
    propagate to NaN velocities. A ~33.3 ms floor is applied on the time
    delta to avoid division blow-ups on duplicated timestamps. Output length
    matches input (a leading 0 is prepended).
    """
    if len(df) < 2:
        return np.zeros(len(df))

    dx = df['x'].diff().values[1:]
    dy = df['y'].diff().values[1:]
    dt = df['timestamp'].diff().values[1:]

    min_dt = 1000.0 / 30  # ~33.3 ms (30 fps floor)
    dt = np.where((dt > 0) | np.isnan(dt), dt, min_dt)

    velocity = np.sqrt(dx ** 2 + dy ** 2) / dt
    return np.concatenate([[0.0], velocity])


def compute_mad(velocity):
    """Median Absolute Deviation of a velocity array (NaN/Inf-safe)."""
    valid = velocity[~np.isnan(velocity) & np.isfinite(velocity)]
    if len(valid) == 0:
        return 0.0
    return float(np.median(np.abs(valid - np.median(valid))))


# ---------------------------------------------------------------------------
# Post-processing of detection output
# ---------------------------------------------------------------------------

def clean_fixations(events_df):
    """Populate ``start_time`` / ``end_time`` / ``event_duration`` per event.

    Recomputes the fixation and saccade event boundaries from per-sample
    timestamps, mapping them back to every sample of the event. Returns the
    DataFrame sorted by timestamp with the index reset.
    """
    events_df.drop(columns=["Unnamed: 4"], inplace=True, errors='ignore')
    events_df['start_time'] = np.nan
    events_df['end_time'] = np.nan
    if 'event_duration' not in events_df.columns:
        events_df['event_duration'] = np.nan

    for id_col in ('fixation_id', 'saccade_id'):
        mask = events_df[id_col].notna()
        if not mask.any():
            continue
        bounds = (
            events_df[mask]
            .groupby(id_col, dropna=False)['timestamp']
            .agg(start_time='min', end_time='max')
        )
        bounds['event_duration'] = bounds['end_time'] - bounds['start_time']
        ids = events_df.loc[mask, id_col]
        events_df.loc[mask, 'start_time']     = ids.map(bounds['start_time'])
        events_df.loc[mask, 'end_time']       = ids.map(bounds['end_time'])
        events_df.loc[mask, 'event_duration'] = ids.map(bounds['event_duration'])

    return events_df.sort_values(['timestamp']).reset_index(drop=True)


def merge_fixations(gaze_data, fixation_merge_threshold):
    """Merge consecutive fixations whose centroids are within the threshold.

    Coordinates of merged fixations are duration-weighted averages.
    Saccade IDs that fall between merged fixations are cleared (call
    :func:`merge_saccades` afterwards to renumber).

    No-op when ``fixation_merge_threshold`` is ``None`` or there is nothing
    to merge.
    """
    if fixation_merge_threshold is None:
        return gaze_data

    fixation_events = (
        gaze_data[gaze_data['event_type'] == 'Fixation']
        .groupby('fixation_id')
        .agg(fixation_x=('fixation_x', 'mean'),
             fixation_y=('fixation_y', 'mean'),
             start_time=('timestamp', 'min'),
             end_time=('timestamp', 'max'))
        .reset_index()
        .rename(columns={'fixation_id': 'event_id'})
        .sort_values('event_id')
    )
    if fixation_events.empty:
        return gaze_data

    merged_logic = []
    current = fixation_events.iloc[0].copy()
    current['merged_ids'] = [current['event_id']]

    for i in range(1, len(fixation_events)):
        nxt = fixation_events.iloc[i]
        distance = np.hypot(current['fixation_x'] - nxt['fixation_x'],
                            current['fixation_y'] - nxt['fixation_y'])
        if distance > fixation_merge_threshold:
            merged_logic.append(current)
            current = nxt.copy()
            current['merged_ids'] = [current['event_id']]
            continue

        # Within threshold: duration-weighted blend
        d_cur = current['end_time'] - current['start_time']
        d_nxt = nxt['end_time'] - nxt['start_time']
        d_total = d_cur + d_nxt
        if d_total > 0:
            current['fixation_x'] = (current['fixation_x'] * d_cur + nxt['fixation_x'] * d_nxt) / d_total
            current['fixation_y'] = (current['fixation_y'] * d_cur + nxt['fixation_y'] * d_nxt) / d_total
        else:
            current['fixation_x'] = (current['fixation_x'] + nxt['fixation_x']) / 2
            current['fixation_y'] = (current['fixation_y'] + nxt['fixation_y']) / 2
        current['end_time'] = nxt['end_time']
        current['merged_ids'].append(nxt['event_id'])

    merged_logic.append(current)

    # Build old_id -> new_id remap
    merged_event_map = {}
    for new_id_zero, fixation in enumerate(merged_logic):
        for old_id in fixation['merged_ids']:
            merged_event_map[old_id] = new_id_zero + 1

    merged_data = gaze_data.copy()
    for fixation in merged_logic:
        new_id = merged_event_map[fixation['merged_ids'][0]]
        for old_id in fixation['merged_ids']:
            mask = merged_data['fixation_id'] == old_id
            if mask.any():
                merged_data.loc[mask, 'fixation_x']  = fixation['fixation_x']
                merged_data.loc[mask, 'fixation_y']  = fixation['fixation_y']
                merged_data.loc[mask, 'fixation_id'] = new_id
                merged_data.loc[mask, 'saccade_id']  = np.nan

    merged_data['merged'] = merged_data['fixation_id'].isin(merged_event_map.values())
    return merged_data


def merge_saccades(events_df):
    """Reassign sequential saccade IDs to consecutive Saccade-runs.

    Should be called after any operation that adds/removes/merges fixations,
    since those operations may leave gaps or stale IDs in the saccade column.
    """
    events_df = events_df.sort_values(by="timestamp").reset_index(drop=True)
    event_type = events_df['event_type'].values
    new_saccade_ids = np.full(len(events_df), np.nan)

    saccade_id_counter = 1
    in_saccade = False
    for i, etype in enumerate(event_type):
        if etype == 'Saccade':
            in_saccade = True
            new_saccade_ids[i] = saccade_id_counter
        elif in_saccade:
            in_saccade = False
            saccade_id_counter += 1

    events_df['saccade_id'] = new_saccade_ids
    return events_df
