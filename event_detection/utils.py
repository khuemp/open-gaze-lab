
import numpy as np
import pandas as pd

def compute_velocity(df):
    """
    Compute velocity for variable framerate data (5-30fps).
    No modifications to original timestamps or data.

    Args:
        df (pd.DataFrame): DataFrame with 'x', 'y', and 'timestamp' columns.

    Returns:
        np.ndarray (np.float): Velocity array (coordinates per second).
    """
    # Ensure no NaNs in x, y, or timestamp
    df_clean = df.dropna(subset=['x', 'y', 'timestamp'])
    
    if len(df_clean) < 2:
        return np.array([])
    
    # Compute differences
    dx = df_clean['x'].diff().values[1:]  # Remove first NaN from diff() (compute dff betwween consecutive points, first point has no previous point)
    dy = df_clean['y'].diff().values[1:]  # Remove first NaN from diff()
    dt = df_clean['timestamp'].diff().values[1:]  # Remove first NaN from diff()
    
    # For variable framerate (5-30fps), minimum time difference should be ~0.033s (30fps)
    # Use this as epsilon to avoid division by zero in velocity calculation
    min_dt = 0.033  # 30fps as minimum reasonable interval
    dt[dt <= 0] = min_dt # Any dt value that is zero or negative is replaced with min_dt.
    
    # Compute velocity (coordinates per second)
    velocity = np.sqrt(dx ** 2 + dy ** 2) / dt 
    
    return velocity


def compute_mad(velocity):
    """
    Compute Median Absolute Deviation (MAD) with robust handling.

    The Median Absolute Deviation (MAD) is a measure of data variability.
    Formula: MAD = median(|velocity_i - median(velocity)|)
    Adaptive Threshold Logic:
        - High MAD: Large velocity variability → More noise → Increase dispersion threshold.
        - Low MAD: Stable velocities → Less noise → Use smaller dispersion threshold.

    Args:
        velocity (np.ndarray): Array of velocity values.

    Returns:
        float: Single MAD value.
    """
    # Remove invalid velocity values
    valid_velocity = velocity[~np.isnan(velocity) & np.isfinite(velocity)]
    
    if len(valid_velocity) == 0:
        return 0.0
    
    # Compute median(velocity)
    median_vel = np.median(valid_velocity)
    
    # Compute |velocity_i - median(velocity)|
    deviations = np.abs(valid_velocity - median_vel)
    
    # Compute MAD = median(|velocity_i - median(velocity)|)
    mad = np.median(deviations)
    
    return mad


def clean_fixations(events_df):
    """
    Cleans and formats the fixation events DataFrame for output.

    Args:
        events_df (pd.DataFrame): DataFrame containing event data with fixations.

    Returns:
        pd.DataFrame: Cleaned DataFrame with renamed columns, start/end frame indices, and one row per fixation.
    """
    events_df.rename(columns={"event_duration": "duration",
                            "fixation_id": "id"}, inplace=True)
    events_df.drop(columns=["Unnamed: 4"], inplace=True, errors='ignore')

    # Compute start/end frame for each fixation_id (exclude null fixation_id)
    fix_bounds = (
        events_df.dropna(subset=['id'])  # ignore rows without a fixation_id
        .groupby('id', dropna=False)['timestamp']
        .agg(start_time='min', end_time='max')
        .reset_index()
    )

    # Merge the start/end back to the original dataframe ---
    events_df = events_df.merge(fix_bounds, on='id', how='left')

    # Drop duplicate fixation_id rows, keep the first occurrence
    # If you want the first occurrence in temporal order, ensure dataframe is sorted by timestamp first:
    events_df = events_df.sort_values(['timestamp']).reset_index(drop=True)

    # Drop duplicates of fixations only (keeps the first row for each fixation_id)
    fixations_df = events_df["event_type"] == "Fixation"
    fixations_unique = (
        events_df[fixations_df]
        .drop_duplicates(subset=["id"], keep="first")
    )
    # Combine back with saccades
    events_df = pd.concat([fixations_unique, events_df[~fixations_df]], ignore_index=True)
    # Keep ordering by timestamp if needed
    events_df = events_df.sort_values(by="timestamp").reset_index(drop=True)

    return events_df


def merge_close_fixations(gaze_data, distance_threshold=100.0):
    """
    Merges consecutive fixations that are close to each other.

    Args:
        gaze_data (pd.DataFrame): DataFrame containing gaze data with event classifications.
        distance_threshold (float): Maximum distance (in pixels) between consecutive fixations to be merged.

    Returns:
        pd.DataFrame: Gaze data with merged fixations (updated fixation coordinates, IDs, and durations).
    """
    # Get gaze points classified as fixations and group by fixation_id to get unique fixations
    fixation_events = gaze_data[gaze_data['event_type'] == 'Fixation'].groupby('fixation_id').agg({
        'fixation_x': 'mean', # average fixation position
        'fixation_y': 'mean',
        'timestamp': ['min', 'max'], # start and end time of fixation
        'fixation_id': 'first' # keep first fixation_id as identifier
    }).reset_index(drop=True)

    # Rename columns
    fixation_events.columns = ['fixation_x', 'fixation_y', 'start_time', 'end_time', 'event_id']

    # Sort by event_id to ensure chronological order
    fixation_events = fixation_events.sort_values('event_id')

    # Create a new DataFrame to store merged fixations
    merged_fixations = []

    # Initialize with the first fixation
    if len(fixation_events) > 0:
        current_fixation = fixation_events.iloc[0].copy()

        # Loop through remaining fixations to check for merging
        for i in range(1, len(fixation_events)):
            next_fixation = fixation_events.iloc[i]

            # Calculate distance between current and next fixation
            distance = ((current_fixation['fixation_x'] - next_fixation['fixation_x']) ** 2 +
                        (current_fixation['fixation_y'] - next_fixation['fixation_y']) ** 2) ** 0.5

            # Check if next fixation is close enough to merge
            if distance <= distance_threshold:
                # Update current fixation with weighted average position and new end time
                total_duration_current = current_fixation['end_time'] - current_fixation['start_time']
                total_duration_next = next_fixation['end_time'] - next_fixation['start_time']
                total_duration = total_duration_current + total_duration_next

                # Calculate weighted average of fixation coordinates
                current_fixation['fixation_x'] = (
                    (current_fixation['fixation_x'] * total_duration_current) +
                    (next_fixation['fixation_x'] * total_duration_next)
                ) / total_duration

                current_fixation['fixation_y'] = (
                    (current_fixation['fixation_y'] * total_duration_current) +
                    (next_fixation['fixation_y'] * total_duration_next)
                ) / total_duration

                # Update end time
                current_fixation['end_time'] = next_fixation['end_time']

                # Track which original event_ids are part of this merged fixation
                if 'merged_ids' not in current_fixation:
                    current_fixation['merged_ids'] = [current_fixation['event_id']]
                current_fixation['merged_ids'].append(next_fixation['event_id'])

            else:
                # Store the current fixation and move to the next
                if 'merged_ids' not in current_fixation:
                    current_fixation['merged_ids'] = [current_fixation['event_id']]
                merged_fixations.append(current_fixation)
                current_fixation = next_fixation.copy()

        # Add the last fixation
        if 'merged_ids' not in current_fixation:
            current_fixation['merged_ids'] = [current_fixation['event_id']]
        merged_fixations.append(current_fixation)

    # If no fixations were found, return the original data
    if not merged_fixations:
        return gaze_data

    # Create new event_id mapping for the merged fixations
    merged_event_map = {}
    for idx, fixation in enumerate(merged_fixations):
        new_event_id = idx + 1
        for old_id in fixation['merged_ids']:
            merged_event_map[old_id] = new_event_id

    # Create a copy of the original data
    merged_data = gaze_data.copy()

    # Assign new fixation coordinates and event_ids
    for old_id, new_id in merged_event_map.items():
        # Find the merged fixation corresponding to this old_id
        merged_fix = [f for f in merged_fixations if old_id in f['merged_ids']][0]

        # Update all rows that were part of this original fixation
        mask = merged_data['fixation_id'] == old_id
        if mask.any():
            merged_data.loc[mask, 'fixation_x'] = merged_fix['fixation_x']
            merged_data.loc[mask, 'fixation_y'] = merged_fix['fixation_y']
            merged_data.loc[mask, 'fixation_id'] = new_id
            merged_data.loc[mask, 'event_duration'] = merged_fix['end_time'] - merged_fix['start_time']

    # Create a new column to track if an event has been merged
    merged_data['merged'] = merged_data['fixation_id'].apply(lambda x: x in merged_event_map.values())
    merged_data['event_id'] = (merged_data['fixation_id'] != merged_data['fixation_id'].shift(1)).cumsum()

    return merged_data

