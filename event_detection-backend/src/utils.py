import numpy as np
import pandas as pd
import logging

def separate_invalid_points(df, resolution):
    """Separates invalid gaze points (NaN and out-of-range) from valid ones.

    Identifies rows where gaze coordinates are NaN or fall outside the screen
    resolution bounds, and splits the DataFrame into valid and invalid subsets.

    Args:
        df (pd.DataFrame): Gaze data with 'x' and 'y' columns in pixel coordinates.
        resolution (tuple): Screen resolution as (width, height) in pixels.

    Returns:
        tuple: (
            pd.DataFrame: Valid gaze points with original index preserved,
            pd.Series: Boolean mask of invalid rows (True = invalid),
            pd.Series: Reason strings for invalid rows ('NaN' or 'Out of Range Gaze Points'),
                       NaN for valid rows
        )
    """
    res_w, res_h = resolution

    is_nan = df['x'].isna() | df['y'].isna()
    is_oor = (
        (df['x'] < 0) | (df['x'] > res_w) |
        (df['y'] < 0) | (df['y'] > res_h)
    ) & ~is_nan  # OOR only for non-NaN rows

    invalid_mask = is_nan | is_oor

    reasons = pd.Series(np.nan, index=df.index, dtype=object)
    reasons[is_nan] = 'NaN'
    reasons[is_oor] = 'Out of Range Gaze Points'

    valid_df = df[~invalid_mask].copy()

    n_nan = is_nan.sum()
    n_oor = is_oor.sum()
    if n_nan > 0 or n_oor > 0:
        logging.info(f"Separated {n_nan} NaN and {n_oor} out-of-range gaze points from {len(df)} total rows")

    return valid_df, invalid_mask, reasons


def reinsert_invalid_points(valid_df, original_length, invalid_mask, reasons):
    """Reinserts invalid gaze points back into the processed DataFrame at original positions.

    Takes the detection results from valid-only data and merges them back with the
    invalid rows, restoring the original row count and order. Invalid rows receive
    their corresponding event_type label ('NaN' or 'Out of Range Gaze Points').

    Args:
        valid_df (pd.DataFrame): Processed detection results for valid gaze points.
        original_length (int): Total number of rows in the original input.
        invalid_mask (pd.Series): Boolean mask where True indicates invalid rows.
        reasons (pd.Series): Event type labels for invalid rows.

    Returns:
        pd.DataFrame: Full-length DataFrame with all rows restored, invalid rows
            having event_type set to their reason and other detection columns as NaN.
    """
    # Build a full-length DataFrame
    full_df = pd.DataFrame(index=range(original_length), columns=valid_df.columns)

    # Place valid results at their original positions
    valid_indices = invalid_mask[~invalid_mask].index
    full_df.loc[valid_indices] = valid_df.values

    # Fill invalid rows with their reason as event_type
    invalid_indices = invalid_mask[invalid_mask].index
    full_df.loc[invalid_indices, 'event_type'] = reasons[invalid_indices].values

    # Preserve numeric dtypes where possible
    for col in valid_df.columns:
        if col == 'event_type':
            continue
        full_df[col] = pd.to_numeric(full_df[col], errors='coerce')

    return full_df


def correct_timestamps(df, sampling_rate):
    """Corrects timestamps to uniform intervals based on sampling rate.

    Regenerates timestamps with fixed intervals calculated from the nominal
    sampling rate, starting from the first timestamp. Useful for correcting
    irregular timestamps caused by recording issues or data loss.

    Args:
        df (pd.DataFrame): Gaze data with timestamp column
        sampling_rate (float): Nominal sampling rate in Hz (e.g., 30, 60, 120)

    Returns:
        pd.DataFrame: Copy of input data with corrected timestamps

    Notes:
        - Interval between samples = 1000 / sampling_rate (in milliseconds)
        - Preserves original first timestamp as reference point
        - Creates uniform temporal spacing regardless of input irregularities
    """
    df_corrected = df.copy()
    interval_ms = 1000.0 / sampling_rate  # Time between samples in ms
    first_timestamp = df_corrected['timestamp'].iloc[0]
    
    # Generate uniform timestamps
    num_samples = len(df_corrected)
    df_corrected['timestamp'] = first_timestamp + np.arange(num_samples) * interval_ms
    
    return df_corrected


def compute_velocity(df):
    """Calculates point-to-point velocities for variable framerate gaze data.

    Computes instantaneous velocities between consecutive gaze points, handling
    variable sampling rates (5-30fps) without modifying original timestamps.
    Uses Euclidean distance for spatial displacement.

    Args:
        df (pd.DataFrame): Gaze data with columns:
            - x (float): X coordinate in pixels
            - y (float): Y coordinate in pixels
            - timestamp (float): Time in milliseconds

    Returns:
        np.ndarray: Array of velocities in pixels per millisecond.
            Same length as input (prepends 0 at start).
            Velocities involving NaN coordinates become NaN.

    Notes:
        - NaN values in x/y coordinates propagate to NaN velocities
        - Enforces minimum time interval of 0.033ms (30fps) to prevent division by zero
        - Returns array of zeros if fewer than 2 points
    """
    if len(df) < 2:
        return np.zeros(len(df))
    
    # Compute differences - NaN values will propagate naturally
    dx = df['x'].diff().values[1:]
    dy = df['y'].diff().values[1:]
    dt = df['timestamp'].diff().values[1:]
    
    # Handle zero or negative time intervals (but preserve NaN)
    min_dt = 0.033  # 30fps as minimum reasonable interval
    dt = np.where((dt > 0) | np.isnan(dt), dt, min_dt)
    
    # Calculate velocity - NaN in dx/dy will produce NaN velocity
    velocity = np.sqrt(dx ** 2 + dy ** 2) / dt
    
    # Prepend 0 at start to match input length (like cateyes does)
    velocity = np.concatenate([[0.], velocity])
    
    return velocity


def compute_mad(velocity):
    """Calculates the Median Absolute Deviation (MAD) of velocity values.

    MAD is a robust measure of variability, less sensitive to outliers than
    standard deviation. Used for adaptive threshold calculations.

    Args:
        velocity (np.ndarray): Array of velocity values.

    Returns:
        float: Single MAD value.
    """
    valid_velocity = velocity[~np.isnan(velocity) & np.isfinite(velocity)]
    
    if len(valid_velocity) == 0:
        return 0.0
    
    median_vel = np.median(valid_velocity)
    deviations = np.abs(valid_velocity - median_vel)
    mad = np.median(deviations)
    
    return mad


def clean_fixations(events_df):
    """Processes and validates fixation data ensuring correct event durations.

    Takes raw event classification results and ensures all fixation-related
    fields are properly populated and event durations are correctly calculated.

    Args:
        events_df (pd.DataFrame): Event data with columns:
            - event_type (str): 'Fixation', 'Saccade', 'NaN', or 'Out of Range Gaze Points'
            - timestamp (float): Time in milliseconds
            - fixation_x (float): X coordinate of fixation center
            - fixation_y (float): Y coordinate of fixation center
            - fixation_id (int): Unique fixation identifier

    Returns:
        pd.DataFrame: Cleaned data with added/corrected columns:
            - event_duration (float): Duration in milliseconds
            All existing columns are preserved with validated values
    """
    # Drop unused columns if they exist
    events_df.drop(columns=["Unnamed: 4"], inplace=True, errors='ignore')
    
    events_df['start_time'] = np.nan
    events_df['end_time'] = np.nan
    
    if 'event_duration' not in events_df.columns:
         events_df['event_duration'] = np.nan

    # Calculate and map fixation bounds
    fix_mask = events_df['fixation_id'].notna()
    if fix_mask.any():
        fix_bounds = (
            events_df[fix_mask]
            .groupby('fixation_id', dropna=False)['timestamp']
            .agg(start_time='min', end_time='max')
        )
        fix_bounds['event_duration'] = fix_bounds['end_time'] - fix_bounds['start_time']
        
        events_df.loc[fix_mask, 'start_time'] = events_df.loc[fix_mask, 'fixation_id'].map(fix_bounds['start_time'])
        events_df.loc[fix_mask, 'end_time'] = events_df.loc[fix_mask, 'fixation_id'].map(fix_bounds['end_time'])
        # Map the newly calculated duration for all fixations
        events_df.loc[fix_mask, 'event_duration'] = events_df.loc[fix_mask, 'fixation_id'].map(fix_bounds['event_duration'])

    # Calculate and map saccade bounds
    sac_mask = events_df['saccade_id'].notna()
    if sac_mask.any():
        sac_bounds = (
            events_df[sac_mask]
            .groupby('saccade_id', dropna=False)['timestamp']
            .agg(start_time='min', end_time='max')
        )
        sac_bounds['event_duration'] = sac_bounds['end_time'] - sac_bounds['start_time']

        events_df.loc[sac_mask, 'start_time'] = events_df.loc[sac_mask, 'saccade_id'].map(sac_bounds['start_time'])
        events_df.loc[sac_mask, 'end_time'] = events_df.loc[sac_mask, 'saccade_id'].map(sac_bounds['end_time'])
        events_df.loc[sac_mask, 'event_duration'] = events_df.loc[sac_mask, 'saccade_id'].map(sac_bounds['event_duration'])

    events_df = events_df.sort_values(['timestamp']).reset_index(drop=True)
    return events_df


def merge_fixations(gaze_data, fixation_merge_threshold=None):
    """Merges spatially and temporally close fixations into single fixation events.

    Identifies consecutive fixations that are within a specified distance threshold
    and combines them into unified fixation events. Uses duration-weighted averaging
    for merged fixation coordinates to maintain gaze behavior representation.

    Args:
        gaze_data (pd.DataFrame): Event classification data with columns:
            - event_type (str): 'Fixation' or 'Saccade'
            - fixation_id (int): Unique fixation identifier
            - fixation_x (float): X coordinate of fixation center
            - fixation_y (float): Y coordinate in pixels
            - timestamp (float): Time in milliseconds
            - saccade_id (int): Unique saccade identifier
        fixation_merge_threshold (float, optional): Maximum distance in pixels
            between fixations to be considered for merging. If None, no
            merging is performed. Defaults to None.

    Returns:
        pd.DataFrame: Updated gaze data with merged fixations, containing:
            - All original columns
            - Updated fixation_id values for merged events
            - Updated fixation_x/y coordinates (duration-weighted averages)
            - New 'merged' column indicating if fixation was combined
            - Preserved saccade events between unmerged fixations

    Notes:
        - Maintains chronological order of events
        - Uses pixel-based distance calculations
        - Weights merged coordinates by fixation durations
        - Falls back to simple averaging if duration calculation fails
        - Returns original data unchanged if no fixations can be merged
        - Clears saccade_ids for merged fixations to maintain consistency
    """
    # Get unique fixation events
    fixation_events = gaze_data[gaze_data['event_type'] == 'Fixation'].groupby('fixation_id').agg({
        'fixation_x': 'mean',
        'fixation_y': 'mean',
        'timestamp': ['min', 'max'],
        'fixation_id': 'first'
    }).reset_index(drop=True)

    # Rename columns
    fixation_events.columns = ['fixation_x', 'fixation_y', 'start_time', 'end_time', 'event_id']

    # Sort by event_id to ensure chronological order
    fixation_events = fixation_events.sort_values('event_id')

    # Create a new DataFrame to store merged fixations
    merged_logic = [] # This list will store the new merged fixation dicts

    # Initialize with the first fixation
    if len(fixation_events) > 0:
        current_fixation = fixation_events.iloc[0].copy()
        
        # Initialize list of merged fixation IDs
        current_fixation['merged_ids'] = [current_fixation['event_id']]

        # Loop through remaining fixations to check for merging
        for i in range(1, len(fixation_events)):
            next_fixation = fixation_events.iloc[i]

            # Calculate distance using PIXEL coordinates
            distance = ((current_fixation['fixation_x'] - next_fixation['fixation_x']) ** 2 +
                       (current_fixation['fixation_y'] - next_fixation['fixation_y']) ** 2) ** 0.5

            if distance <= fixation_merge_threshold:
                # Calculate duration-weighted average position
                total_duration_current = current_fixation['end_time'] - current_fixation['start_time']
                total_duration_next = next_fixation['end_time'] - next_fixation['start_time']
                total_duration = total_duration_current + total_duration_next

                if total_duration > 0:
                    current_fixation['fixation_x'] = (
                        (current_fixation['fixation_x'] * total_duration_current) +
                        (next_fixation['fixation_x'] * total_duration_next)
                    ) / total_duration
                    current_fixation['fixation_y'] = (
                        (current_fixation['fixation_y'] * total_duration_current) +
                        (next_fixation['fixation_y'] * total_duration_next)
                    ) / total_duration
                else:
                    current_fixation['fixation_x'] = (current_fixation['fixation_x'] + next_fixation['fixation_x']) / 2
                    current_fixation['fixation_y'] = (current_fixation['fixation_y'] + next_fixation['fixation_y']) / 2
                
                current_fixation['end_time'] = next_fixation['end_time']
                
                # Add the merged fixation ID to tracking list
                current_fixation['merged_ids'].append(next_fixation['event_id'])

            else:
                # Store the completed fixation
                merged_logic.append(current_fixation)
                # Start a new fixation
                current_fixation = next_fixation.copy()
                current_fixation['merged_ids'] = [current_fixation['event_id']] # <--- CREATE 'merged_ids' for new fix

        # Add the very last fixation
        merged_logic.append(current_fixation)

    # If no fixations were found, return the original data
    if not merged_logic:
        return gaze_data

    merged_event_map = {}
    for idx, fixation in enumerate(merged_logic):
        new_event_id = idx + 1

        for old_id in fixation['merged_ids']:
            merged_event_map[old_id] = new_event_id

    # Create a copy of the original data
    merged_data = gaze_data.copy()

    # Assign new fixation coordinates and event_ids
    for old_id, new_id in merged_event_map.items():
        # Find the corresponding merged fixation
        merged_fix = [f for f in merged_logic if old_id in f['merged_ids']][0]

        # Update all rows that were part of this original fixation
        mask = merged_data['fixation_id'] == old_id
        if mask.any():
            merged_data.loc[mask, 'fixation_x'] = merged_fix['fixation_x']
            merged_data.loc[mask, 'fixation_y'] = merged_fix['fixation_y']
            merged_data.loc[mask, 'fixation_id'] = new_id
            merged_data.loc[mask, 'saccade_id'] = np.nan
    # Create a new column to track if an event has been merged
    merged_data['merged'] = merged_data['fixation_id'].apply(lambda x: x in merged_event_map.values())

    return merged_data

def merge_saccades(events_df):
    """Updates saccade IDs after fixation merging or other modifications.

    Ensures saccade events between fixations have correct, sequential IDs.
    Should be called after any operation that modifies fixation classifications
    or merges fixations.

    Args:
        events_df (pd.DataFrame): Event data with columns:
            - event_type (str): 'Fixation' or 'Saccade'
            - fixation_id (int): Unique fixation identifier
            - saccade_id (int): Unique saccade identifier, may be NaN

    Returns:
        pd.DataFrame: Updated data with:
            - Reassigned sequential saccade IDs
            - All other columns preserved unchanged
    """
    events_df = events_df.sort_values(by="timestamp").reset_index(drop=True)

    event_type = events_df['event_type'].values
    new_saccade_ids = np.full(len(events_df), np.nan) 

    saccade_id_counter = 1
    in_saccade = False

    for i in range(len(event_type)):
        if event_type[i] == 'Saccade':
            if not in_saccade:
                in_saccade = True
            new_saccade_ids[i] = saccade_id_counter
        elif in_saccade: 
            in_saccade = False
            saccade_id_counter += 1

    events_df['saccade_id'] = new_saccade_ids
    
    return events_df

