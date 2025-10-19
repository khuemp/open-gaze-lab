
import numpy as np
import pandas as pd
from .utils import compute_velocity, compute_mad


def prepare_classification_data(gaze_data, threshold, adapt=False, is_velocity_based=True):
    """
    Helper to prepare gaze data, scale coordinates, compute velocity, optionally adapt threshold,
    and allocate result arrays for fixation classification algorithms.
    """
    # Work with a copy to avoid modifying original data
    result_data = gaze_data.copy()
    
    # Scale coordinates to pixels for dispersion/velocity calculation
    x = result_data['x'].values
    y = result_data['y'].values
    t = result_data['timestamp'].values

    # Number of gaze points
    n = len(x)
    # Initialize event_type = 'Saccade', norm_pos_x = NaN, norm_pos_y = NaN, event_duration = NaN, fixation_id = 0
    event_type = np.full(n, 'Saccade', dtype='U10')
    norm_pos_x = np.full(n, np.nan)
    norm_pos_y = np.full(n, np.nan)
    event_duration = np.full(n, np.nan)
    fixation_ids = np.full(n, np.nan) # Use NaN for non-fixation points
    saccade_ids = np.full(n, np.nan)  # Use NaN for non-saccade points

    # Adaptive threshold computation
    original_threshold = threshold
    
    # Create temporary dataframe with scaled coordinates for velocity computation
    temp_df = pd.DataFrame({'x': x, 'y': y, 'timestamp': t})
    velocity = compute_velocity(temp_df)

    # If adapt is True, compute velocity and MAD to adjust dispersion/velocity threshold
    if adapt:
        if len(velocity) > 0:
            mad_velocity = compute_mad(velocity)
            if mad_velocity > 0:
                # Adapt threshold: higher MAD = more noise = higher threshold
                alpha = 0.1  # Tuning parameter
                adaptation_factor = 1 + alpha * mad_velocity
                threshold = original_threshold * adaptation_factor

                metric = "velocity" if is_velocity_based else "dispersion"
                print(f"Original {metric} threshold: {original_threshold:.2f}, "
                      f"MAD velocity: {mad_velocity:.4f}, "
                      f"Adaptive threshold: {threshold:.2f}")
            else:
                print(f"Using original threshold: {threshold} (MAD = 0)")
        else:
            print(f"Using original threshold: {threshold} (no valid velocity)")

    return result_data, x, y, t, n, threshold, velocity, event_type, norm_pos_x, norm_pos_y, event_duration, fixation_ids, saccade_ids


def finalize_result_dataframe(result_data, event_type, norm_pos_x, norm_pos_y, event_duration, fixation_ids, saccade_ids):
    """Assign additional columns with computed values to the DataFrame and return it."""
    result_data['event_type'] = event_type
    result_data['norm_pos_x'] = norm_pos_x
    result_data['norm_pos_y'] = norm_pos_y
    result_data['event_duration'] = event_duration
    result_data['fixation_id'] = fixation_ids
    result_data['saccade_id'] = saccade_ids
    return result_data

def add_saccade_ids(event_type, saccade_ids):
    """Helper function to find consecutive saccades and assign them a unique ID."""
    saccade_id_counter = 1 # Saccade counter starts at 1
    in_saccade = False
    for i in range(len(event_type)):
        if event_type[i] == 'Saccade':
            if not in_saccade:
                in_saccade = True
            saccade_ids[i] = saccade_id_counter
        elif in_saccade: # End of a saccade block
            in_saccade = False
            saccade_id_counter += 1 # Increment only when a new saccade block begins
    return saccade_ids



def classify_idt(gaze_data, dispersion_threshold=100.0, min_fixation_duration=50.0/1000, adapt=False):
    """
    Classifies gaze points into fixations and saccades using the I-DT algorithm.
    I-DT computes position of points in space and classifies points that are close together (low dispersion) as fixations.
    Works with variable framerate data without modifying original timestamps.

    Args:
        gaze_data (pd.DataFrame): DataFrame containing gaze data with 'x', 'y', and 'timestamp' columns.
        dispersion_threshold (float): Maximum allowed dispersion within a fixation window (in pixels).
        min_fixation_duration (float): Minimum duration (in seconds) for a fixation to be considered valid.
        adapt (bool): Whether to adapt the dispersion threshold based on velocity.

    Returns:
        pd.DataFrame: Gaze data with added 'event_type', 'norm_pos_x', 'norm_pos_y', and 'event_duration' columns.
    """
    (result_data, x, y, t, n, dispersion_threshold, velocity,
     event_type, norm_pos_x, norm_pos_y, event_duration, fixation_ids, saccade_ids) = \
        prepare_classification_data(gaze_data, dispersion_threshold, adapt, is_velocity_based=False)

    start_idx = 0
    fixation_id = 1

    while start_idx < n:
        current_idx = start_idx
        max_x = x[start_idx]
        min_x = x[start_idx]
        max_y = y[start_idx]
        min_y = y[start_idx]

        # Expand window for next gaze points until dispersion exceeds threshold
        while current_idx < n:
            current_x = x[current_idx]
            current_y = y[current_idx]

            max_x = max(max_x, current_x)
            min_x = min(min_x, current_x)
            max_y = max(max_y, current_y)
            min_y = min(min_y, current_y)

            dispersion = (max_x - min_x) + (max_y - min_y)

            if dispersion > dispersion_threshold:
                break

            # Numer of points in the current window
            current_idx += 1

        # Calculate duration of the current window
        end_idx = current_idx - 1 if current_idx > start_idx else start_idx
        window_duration = t[end_idx] - t[start_idx] if end_idx > start_idx else 0.0

        # Check if window meets minimum fixation duration
        if window_duration >= min_fixation_duration:
            # Classify as fixation
            event_type[start_idx:end_idx + 1] = 'Fixation'
            
            # Calculate fixation center in normalized coordinates
            fix_x = np.mean(x[start_idx:end_idx + 1])
            fix_y = np.mean(y[start_idx:end_idx + 1])
            
            # Assign fixation coordinates and duration to arrays that store results
            norm_pos_x[start_idx:end_idx + 1] = fix_x
            norm_pos_y[start_idx:end_idx + 1] = fix_y
            event_duration[start_idx:end_idx + 1] = window_duration
            fixation_ids[start_idx:end_idx + 1] = fixation_id
            fixation_id += 1

        start_idx = end_idx + 1 if current_idx > start_idx else start_idx + 1

    saccade_ids = add_saccade_ids(event_type, saccade_ids)

    return finalize_result_dataframe(result_data, event_type, norm_pos_x, norm_pos_y, event_duration, fixation_ids, saccade_ids)


def classify_ivt(gaze_data, velocity_threshold=100.0, min_fixation_duration=50.0/1000, adapt=False):
    """
    Classifies gaze points into fixations and saccades using the I-VT algorithm.
    I-VT computes point-to-point velocities and classifies points as fixations if velocity is below a threshold.
    Works with variable framerate data without modifying original timestamps.

    Args:
        gaze_data (pd.DataFrame): Input gaze data with columns 'x','y','timestamp'. x,y assumed normalized [0,1].
        velocity_threshold (float): Maximum allowed velocity (pixels/second) for points considered in a fixation.
        min_fixation_duration (float): Minimum fixation duration in seconds.
        adapt (bool): If True, adapt the velocity_threshold based on MAD of velocities.

    Returns:
        pd.DataFrame: DataFrame with added columns: 'event_type','norm_pos_x','norm_pos_y','event_duration','fixation_id'.
    """
    (result_data, x, y, t, n, velocity_threshold, velocity,
     event_type, norm_pos_x, norm_pos_y, event_duration, fixation_ids, saccade_ids) = \
        prepare_classification_data(gaze_data, velocity_threshold, adapt, is_velocity_based=True)

    start_idx = 0
    fixation_id = 1
    
    # Expand velocity array so it matches number of gaze points
    if len(velocity) > 0:
        velocity = np.insert(velocity, 0, 0.0)  # set first velocity to 0 for first gaze point

    while start_idx < n:
        current_idx = start_idx
        # Expand window while velocity is below threshold 
        while current_idx < n:
            if velocity[current_idx] > velocity_threshold:
                break
            current_idx += 1

        # Calculate duration of the current low-velocity window
        end_idx = current_idx - 1 if current_idx > start_idx else start_idx
        window_duration = t[end_idx] - t[start_idx] if end_idx > start_idx else 0.0

        # Check if window meets minimum fixation duration
        if window_duration >= min_fixation_duration:
            event_type[start_idx:end_idx + 1] = 'Fixation'

            # Calculate fixation center
            fix_x = np.mean(x[start_idx:end_idx + 1])
            fix_y = np.mean(y[start_idx:end_idx + 1])

            norm_pos_x[start_idx:end_idx + 1] = fix_x
            norm_pos_y[start_idx:end_idx + 1] = fix_y
            event_duration[start_idx:end_idx + 1] = window_duration
            fixation_ids[start_idx:end_idx + 1] = fixation_id
            fixation_id += 1

        # Move to next segment
        start_idx = end_idx + 1

    saccade_ids = add_saccade_ids(event_type, saccade_ids)

    return finalize_result_dataframe(result_data, event_type, norm_pos_x, norm_pos_y, event_duration, fixation_ids, saccade_ids)
