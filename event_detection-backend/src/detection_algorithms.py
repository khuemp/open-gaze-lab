
import numpy as np
import pandas as pd
from .utils import compute_velocity, compute_mad


def prepare_classification_data(gaze_data: pd.DataFrame, 
                          threshold: float, 
                          adapt: bool = False, 
                          tuning_parameter: float = 0.1, 
                          is_velocity_based: bool = True) -> tuple:
    """Prepares gaze data for fixation classification algorithms.

    Prepares a copy of the gaze data, scales coordinates to pixels, computes velocity,
    optionally adapts the threshold based on movement patterns, and allocates result arrays.

    Args:
        gaze_data (pd.DataFrame): Input gaze data with required columns:
            - x (float): X coordinate
            - y (float): Y coordinate
            - timestamp (float): Time in milliseconds
        threshold (float): Initial detection threshold in pixels
        adapt (bool, optional): Whether to adapt threshold. Defaults to False.
        tuning_parameter (float, optional): Adaptation strength factor. Defaults to 0.1.
        is_velocity_based (bool, optional): True for IVT, False for IDT. Defaults to True.

    Returns:
        tuple: (
            pd.DataFrame: Copy of input data,
            np.ndarray: X coordinates array,
            np.ndarray: Y coordinates array,
            np.ndarray: Timestamps array,
            np.ndarray: Event type labels array (initialized as 'Saccade'),
            np.ndarray: Fixation X coordinates array (initialized as NaN),
            np.ndarray: Fixation Y coordinates array (initialized as NaN),
            np.ndarray: Event durations array (initialized as NaN),
            np.ndarray: Fixation IDs array (initialized as NaN),
            np.ndarray: Saccade IDs array (initialized as NaN),
            float: Final threshold after adaptation
        )
    """
    # Create independent copy of input data
    result_data = gaze_data.copy()
    
    # Extract coordinate arrays for efficient processing
    x = result_data['x'].values
    y = result_data['y'].values
    t = result_data['timestamp'].values

    # Initialize data structures for event detection
    n = len(x)
    event_type = np.full(n, 'Saccade', dtype='U10')        # Default label is saccade
    fixation_x = np.full(n, np.nan)                        # Fixation center X
    fixation_y = np.full(n, np.nan)                        # Fixation center Y
    event_duration = np.full(n, np.nan)                    # Duration of each event
    fixation_ids = np.full(n, np.nan)                      # Unique fixation identifiers
    saccade_ids = np.full(n, np.nan)                      # Unique saccade identifiers

    # Store initial threshold for potential adaptation
    original_threshold = threshold
    
    # Compute point-to-point velocities for threshold adaptation
    temp_df = pd.DataFrame({'x': x, 'y': y, 'timestamp': t})
    velocity = compute_velocity(temp_df)

    # Adapt threshold based on movement variability
    if adapt:
        if len(velocity) > 0:
            mad_velocity = compute_mad(velocity)
            if mad_velocity > 0:
                # Scale threshold by movement noise level
                adaptation_factor = 1 + tuning_parameter * mad_velocity
                threshold = original_threshold * adaptation_factor

                metric = "velocity" if is_velocity_based else "dispersion"
                print(f"Original {metric} threshold: {original_threshold:.2f}, "
                      f"MAD velocity: {mad_velocity:.4f}, "
                      f"Adaptive threshold: {threshold:.2f}")
            else:
                print(f"Using original threshold: {threshold} (MAD = 0)")
        else:
            print(f"Using original threshold: {threshold} (no valid velocity)")

    return result_data, x, y, t, n, threshold, velocity, event_type, fixation_x, fixation_y, event_duration, fixation_ids, saccade_ids


def finalize_result_dataframe(result_data, event_type, fixation_x, fixation_y,
                         event_duration, fixation_ids, saccade_ids):
    """Finalizes classification results by adding computed values to the DataFrame.

    Args:
        result_data (pd.DataFrame): Original gaze data DataFrame
        event_type (np.ndarray): Array of event classifications ('Fixation' or 'Saccade')
        fixation_x (np.ndarray): X coordinates of fixation centers
        fixation_y (np.ndarray): Y coordinates of fixation centers
        event_duration (np.ndarray): Duration of each event in milliseconds
        fixation_ids (np.ndarray): Unique IDs for fixations
        saccade_ids (np.ndarray): Unique IDs for saccades

    Returns:
        pd.DataFrame: Input DataFrame with added columns:
            - event_type (str): Classification label
            - fixation_x (float): X coordinate of fixation
            - fixation_y (float): Y coordinate of fixation
            - event_duration (float): Duration in milliseconds
            - fixation_id (int): Unique fixation identifier
            - saccade_id (int): Unique saccade identifier
    """
    result_data['event_type'] = event_type
    result_data['fixation_x'] = fixation_x
    result_data['fixation_y'] = fixation_y
    result_data['event_duration'] = event_duration
    result_data['fixation_id'] = fixation_ids
    result_data['saccade_id'] = saccade_ids
    return result_data

def add_saccade_ids(event_type, saccade_ids):
    """Assigns unique IDs to consecutive sequences of saccade points.

    Args:
        event_type (np.ndarray): Array of event classifications ('Fixation' or 'Saccade')
        saccade_ids (np.ndarray): Array to store saccade IDs, initialized with NaN

    Returns:
        np.ndarray: Updated array with consecutive integers assigned to each sequence of saccade points
    """
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



def classify_idt(gaze_data, dispersion_threshold=150.0, min_fixation_duration=50, adapt=False, tuning_parameter=0.1):
    """
    Classifies gaze points into fixations and saccades using the I-DT algorithm.
    I-DT computes position of points in space and classifies points that are close together (low dispersion) as fixations.
    Works with variable framerate data without modifying original timestamps.

    Args:
        gaze_data (pd.DataFrame): DataFrame containing gaze data with 'x', 'y', and 'timestamp' columns.
        dispersion_threshold (float): Maximum allowed dispersion within a fixation window (in pixels).
        min_fixation_duration (float): Minimum duration (in milliseconds) for a fixation to be considered valid.
        adapt (bool): Whether to adapt the dispersion threshold based on velocity.

    Returns:
        pd.DataFrame: Gaze data with added 'event_type', 'fixation_x', 'fixation_y', and 'event_duration' columns.
    """
    (result_data, x, y, t, n, dispersion_threshold, velocity,
     event_type, fixation_x, fixation_y, event_duration, fixation_ids, saccade_ids) = \
        prepare_classification_data(gaze_data, dispersion_threshold, adapt, tuning_parameter, is_velocity_based=False)

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
            # Mark window points as a fixation event
            event_type[start_idx:end_idx + 1] = 'Fixation'
            
            # Calculate centroid of fixation points
            fix_x = np.mean(x[start_idx:end_idx + 1])
            fix_y = np.mean(y[start_idx:end_idx + 1])
            
            # Record fixation properties for all points in window
            fixation_x[start_idx:end_idx + 1] = fix_x
            fixation_y[start_idx:end_idx + 1] = fix_y
            event_duration[start_idx:end_idx + 1] = window_duration
            fixation_ids[start_idx:end_idx + 1] = fixation_id
            fixation_id += 1

        start_idx = end_idx + 1 if current_idx > start_idx else start_idx + 1

    saccade_ids = add_saccade_ids(event_type, saccade_ids)

    return finalize_result_dataframe(result_data, event_type, fixation_x, fixation_y, event_duration, fixation_ids, saccade_ids)


def classify_ivt(gaze_data, velocity_threshold=150.0, min_fixation_duration=50,
               adapt=False, tuning_parameter=0.1):
    """Identifies fixations using the I-VT (Velocity Threshold) algorithm.

    Classifies gaze points as fixations when their velocity is below a threshold for a
    minimum duration. Uses point-to-point velocities and handles variable sampling rates
    by using actual timestamps. Can adaptively adjust the threshold based on movement patterns.

    Args:
        gaze_data (pd.DataFrame): Input data with required columns:
            - x (float): X coordinate in pixels
            - y (float): Y coordinate in pixels
            - timestamp (float): Time in milliseconds
        velocity_threshold (float, optional): Max velocity in pixels/ms. Defaults to 150.0.
        min_fixation_duration (int, optional): Minimum fixation time in ms. Defaults to 50.
        adapt (bool, optional): Whether to adapt threshold. Defaults to False.
        tuning_parameter (float, optional): Adaptation strength factor. Defaults to 0.1.

    Returns:
        pd.DataFrame: Original data with added columns:
            - event_type (str): 'Fixation' or 'Saccade'
            - fixation_x (float): X coordinate of fixation center
            - fixation_y (float): Y coordinate of fixation center
            - event_duration (float): Duration in milliseconds
            - fixation_id (int): Unique fixation identifier
            - saccade_id (int): Unique saccade identifier
    """
    (result_data, x, y, t, n, velocity_threshold, velocity,
     event_type, fixation_x, fixation_y, event_duration, fixation_ids, saccade_ids) = \
        prepare_classification_data(gaze_data, velocity_threshold, adapt, tuning_parameter,is_velocity_based=True)

    start_idx = 0
    fixation_id = 1

    while start_idx < n:
        current_idx = start_idx
        # Find consecutive points with velocity under threshold
        while current_idx < n:
            if velocity[current_idx] > velocity_threshold:
                break
            current_idx += 1

        # Measure temporal span of low-velocity window
        end_idx = current_idx - 1 if current_idx > start_idx else start_idx
        window_duration = t[end_idx] - t[start_idx] if end_idx > start_idx else 0.0

        # Validate potential fixation duration
        if window_duration >= min_fixation_duration:
            # Mark points as part of fixation event
            event_type[start_idx:end_idx + 1] = 'Fixation'

            # Compute centroid of fixation points
            fix_x = np.mean(x[start_idx:end_idx + 1])
            fix_y = np.mean(y[start_idx:end_idx + 1])

            # Record fixation properties for all points in window
            fixation_x[start_idx:end_idx + 1] = fix_x
            fixation_y[start_idx:end_idx + 1] = fix_y
            event_duration[start_idx:end_idx + 1] = window_duration
            fixation_ids[start_idx:end_idx + 1] = fixation_id
            fixation_id += 1

        # Advance to next potential fixation window
        start_idx = end_idx + 1

    saccade_ids = add_saccade_ids(event_type, saccade_ids)

    return finalize_result_dataframe(result_data, event_type, fixation_x, fixation_y, event_duration, fixation_ids, saccade_ids)
