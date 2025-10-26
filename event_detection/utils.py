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
    dx = df_clean['x'].diff().values[1:]
    dy = df_clean['y'].diff().values[1:]
    dt = df_clean['timestamp'].diff().values[1:]
    
    min_dt = 0.033  # 30fps as minimum reasonable interval
    dt[dt <= 0] = min_dt 
    
    velocity = np.sqrt(dx ** 2 + dy ** 2) / dt 
    
    return velocity


def compute_mad(velocity):
    """
    Compute Median Absolute Deviation (MAD) with robust handling.

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
    """
    Cleans and formats the event DataFrame by calculating start_time, 
    end_time, and duration for all events (fixations and saccades)
    and mapping these values back to *every gaze point* row.
    
    This function does NOT summarize the data and preserves the 
    one-row-per-gaze-point format for all scenarios.

    Args:
        events_df (pd.DataFrame): DataFrame containing event data (one row per gaze point).

    Returns:
        pd.DataFrame: Cleaned DataFrame with start/end/duration mapped to every row.
    """
    # Drop unused columns if they exist
    events_df.drop(columns=["Unnamed: 4"], inplace=True, errors='ignore')
    
    events_df['start_time'] = np.nan
    events_df['end_time'] = np.nan
    
    if 'duration' not in events_df.columns:
         events_df['duration'] = np.nan

    # --- 1. Calculate and Map Fixation Bounds ---
    fix_mask = events_df['fixation_id'].notna()
    if fix_mask.any():
        fix_bounds = (
            events_df[fix_mask]
            .groupby('fixation_id', dropna=False)['timestamp']
            .agg(start_time='min', end_time='max')
        )
        fix_bounds['duration'] = fix_bounds['end_time'] - fix_bounds['start_time']
        
        events_df.loc[fix_mask, 'start_time'] = events_df.loc[fix_mask, 'fixation_id'].map(fix_bounds['start_time'])
        events_df.loc[fix_mask, 'end_time'] = events_df.loc[fix_mask, 'fixation_id'].map(fix_bounds['end_time'])
        # Map the newly calculated duration for all fixations
        events_df.loc[fix_mask, 'duration'] = events_df.loc[fix_mask, 'fixation_id'].map(fix_bounds['duration'])

    # --- 2. Calculate and Map Saccade Bounds ---
    sac_mask = events_df['saccade_id'].notna()
    if sac_mask.any():
        sac_bounds = (
            events_df[sac_mask]
            .groupby('saccade_id', dropna=False)['timestamp']
            .agg(start_time='min', end_time='max')
        )
        sac_bounds['duration'] = sac_bounds['end_time'] - sac_bounds['start_time']

        events_df.loc[sac_mask, 'start_time'] = events_df.loc[sac_mask, 'saccade_id'].map(sac_bounds['start_time'])
        events_df.loc[sac_mask, 'end_time'] = events_df.loc[sac_mask, 'saccade_id'].map(sac_bounds['end_time'])
        events_df.loc[sac_mask, 'duration'] = events_df.loc[sac_mask, 'saccade_id'].map(sac_bounds['duration'])

    events_df.rename(columns={"event_duration": "original_gaze_point_duration"}, inplace=True, errors='ignore')
    
    events_df = events_df.sort_values(['timestamp']).reset_index(drop=True)
    return events_df


def merge_fixations(gaze_data, fixation_merge_threshold=100.0):
    """
    Merges consecutive fixations that are close, re-classifying all gaze points
    between them (including saccades) as part of the new, larger fixation event.
    """
    fixation_events = gaze_data[gaze_data['event_type'] == 'Fixation'].groupby('fixation_id').agg(
        fixation_x=('fixation_x', 'mean'),
        fixation_y=('fixation_y', 'mean'),
        start_time=('timestamp', 'min'),
        end_time=('timestamp', 'max')
    ).reset_index().rename(columns={"fixation_id": "event_id"})
    
    fixation_events = fixation_events.sort_values('start_time')

    if len(fixation_events) < 2:
        return gaze_data

    merged_logic = []
    current_fix = fixation_events.iloc[0].to_dict()

    for i in range(1, len(fixation_events)):
        next_fix = fixation_events.iloc[i].to_dict()
        
        distance = ((current_fix['fixation_x'] - next_fix['fixation_x']) ** 2 +
                    (current_fix['fixation_y'] - next_fix['fixation_y']) ** 2) ** 0.5

        if distance <= fixation_merge_threshold:
            total_duration_current = current_fix['end_time'] - current_fix['start_time']
            total_duration_next = next_fix['end_time'] - next_fix['start_time']
            total_duration = total_duration_current + total_duration_next

            if total_duration > 0:
                current_fix['fixation_x'] = ((current_fix['fixation_x'] * total_duration_current) + (next_fix['fixation_x'] * total_duration_next)) / total_duration
                current_fix['fixation_y'] = ((current_fix['fixation_y'] * total_duration_current) + (next_fix['fixation_y'] * total_duration_next)) / total_duration
            else:
                current_fix['fixation_x'] = (current_fix['fixation_x'] + next_fix['fixation_x']) / 2
                current_fix['fixation_y'] = (current_fix['fixation_y'] + next_fix['fixation_y']) / 2
            
            current_fix['end_time'] = next_fix['end_time']
        
        else:
            merged_logic.append(current_fix)
            current_fix = next_fix
    
    merged_logic.append(current_fix)

    merged_data = gaze_data.copy()
    
    new_fixation_id_counter = 1
    for merged_event in merged_logic:
        new_id = new_fixation_id_counter
        
        min_start_time = merged_event['start_time']
        max_end_time = merged_event['end_time']
        
        new_x = merged_event['fixation_x']
        new_y = merged_event['fixation_y']
        
        mask = (merged_data['timestamp'] >= min_start_time) & (merged_data['timestamp'] <= max_end_time)
        
        merged_data.loc[mask, 'event_type'] = 'Fixation'
        merged_data.loc[mask, 'fixation_id'] = new_id
        merged_data.loc[mask, 'saccade_id'] = np.nan
        merged_data.loc[mask, 'fixation_x'] = new_x
        merged_data.loc[mask, 'fixation_y'] = new_y
        
        new_fixation_id_counter += 1

    return merged_data


def merge_saccades(events_df):
    """
    Re-indexes saccade events. Designed to be run AFTER merge_fixations
    to combine any saccades that are now adjacent.
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

