
import numpy as np
import pandas as pd
from .utils import compute_velocity, compute_mad
from .preprocessing import preprocess_gaze_data


def prepare_classification_data(gaze_data: pd.DataFrame, 
                          threshold: float, 
                          adapt: bool = False, 
                          tuning_parameter: float = 0.1, 
                          is_velocity_based: bool = True,
                          sampling_rate: float = None) -> tuple:
    """Prepares gaze data for fixation classification algorithms.

    When a *sampling_rate* is provided the enhanced I-VAT+Frel preprocessing
    pipeline is applied:

    1. **Savitzky-Golay smoothing** – produces ``filter_x`` / ``filter_y``.
    2. **Gaze velocity** computed on smoothed coordinates using the average
       time-delta (lower noise than per-sample Δt).
    3. **Optical-flow compensation** (if ``flow_x`` / ``flow_y`` columns are
       present in the input) – subtracts background motion to yield
       ``vel_rel_mag``.
    4. **Adaptive threshold from flow RMS** (if ``adapt=True`` *and* flow
       columns are present) – ``threshold_i = base + gain × FlowRMS_i``.
    5. **MAD-based adaptive threshold** as fallback when no flow columns are
       available and ``adapt=True``.

    If *sampling_rate* is ``None`` the function falls back to the original
    point-to-point velocity computation so that existing behaviour is fully
    preserved.

    Args:
        gaze_data: Input gaze data with ``x``, ``y``, ``timestamp`` columns
            (and optionally ``flow_x``, ``flow_y``).
        threshold: Initial detection threshold (px/ms for I-VT, px for I-DT).
        adapt: Whether to adapt the threshold.
        tuning_parameter: Strength factor for the MAD-based fallback
            adaptation (used only when no flow data is available).
        is_velocity_based: ``True`` for I-VT, ``False`` for I-DT.
        sampling_rate: Recording sampling rate in Hz.  When provided the
            enhanced Savgol + flow pipeline is activated.

    Returns:
        A tuple of:
            - result_data (DataFrame)
            - x, y, t (ndarrays)
            - n (int) – number of samples
            - threshold (float) – possibly adapted global threshold
            - velocity (ndarray) – per-sample velocity for the legacy
              windowing loop
            - event_type, fixation_x, fixation_y, event_duration,
              fixation_ids, saccade_ids (ndarrays)
            - preprocess_meta (dict | None) – metadata from the enhanced
              pipeline (``None`` when *sampling_rate* is not given)
    """
    # Create independent copy of input data
    result_data = gaze_data.copy()

    # ------------------------------------------------------------------
    # Enhanced I-VAT+Frel preprocessing – only activated when optical-flow
    # columns are present so that non-head-mounted data keeps the exact
    # same legacy behaviour.
    # ------------------------------------------------------------------
    preprocess_meta = None
    has_flow = "flow_x" in result_data.columns and "flow_y" in result_data.columns
    if sampling_rate is not None and sampling_rate > 0 and has_flow:
        preprocess_meta = preprocess_gaze_data(
            result_data,
            sampling_rate,
            is_velocity_based=is_velocity_based,
            base_threshold=threshold,
            adapt=adapt,
            # Paper defaults for gain and flow-RMS window
            gain=1.0,
            window_size_ms=500.0,
            savgol_window_ms=55.0,
        )

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
    saccade_ids = np.full(n, np.nan)                       # Unique saccade identifiers

    # Store initial threshold for potential adaptation
    original_threshold = threshold

    # ------------------------------------------------------------------
    # Build the velocity array used by the windowing loop.
    # When enhanced preprocessing ran, we use the (possibly flow-relative)
    # velocity column it computed.  Otherwise fall back to the legacy
    # point-to-point velocity helper.
    # ------------------------------------------------------------------
    if preprocess_meta is not None and is_velocity_based:
        vel_col = preprocess_meta["vel_col"]  # "vel_rel_mag" or "vel_mag"
        velocity = result_data[vel_col].fillna(0).values
    else:
        temp_df = pd.DataFrame({'x': x, 'y': y, 'timestamp': t})
        velocity = compute_velocity(temp_df)

    # ------------------------------------------------------------------
    # Adapt threshold (MAD fallback – only when enhanced pipeline did NOT
    # already create a per-sample adaptive threshold column).
    # ------------------------------------------------------------------
    if adapt and (preprocess_meta is None or not preprocess_meta.get("has_adaptive_threshold", False)):
        if len(velocity) > 0:
            mad_velocity = compute_mad(velocity)
            if mad_velocity > 0:
                adaptation_factor = 1 + tuning_parameter * mad_velocity
                threshold = original_threshold * adaptation_factor

    return result_data, x, y, t, n, threshold, velocity, event_type, fixation_x, fixation_y, event_duration, fixation_ids, saccade_ids, preprocess_meta


def finalize_result_dataframe(result_data, event_type, fixation_x, fixation_y,
                         event_duration, fixation_ids, saccade_ids):
    """Finalizes classification results by adding computed values to the DataFrame.

    Args:
        result_data (pd.DataFrame): Original gaze data DataFrame
        event_type (np.ndarray): Array of event classifications ('Fixation' or 'Saccade')
        fixation_x (np.ndarray): X coordinates of fixation events
        fixation_y (np.ndarray): Y coordinates of fixation events
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
    """Assigns unique IDs to consecutive sequences of saccade samples.

    Args:
        event_type (np.ndarray): Array of event classifications ('Fixation' or 'Saccade')
        saccade_ids (np.ndarray): Array to store saccade IDs, initialized with NaN

    Returns:
        np.ndarray: Updated array with consecutive integers assigned to each sequence of saccade samples
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



def classify_idt(gaze_data, dispersion_threshold=150.0, min_fixation_duration=50,
                 adapt=False, tuning_parameter=0.1, sampling_rate=None):
    """Classifies gaze samples into fixations and saccades using the I-DT algorithm.

    Enhanced with the I-VAT+Frel pipeline when *sampling_rate* is provided:

    * **Savitzky-Golay smoothing** reduces coordinate noise before dispersion
      is computed.
    * **Relative dispersion** (flow-compensated) is used when ``flow_x`` /
      ``flow_y`` columns are present — this is the *Frel* head-motion
      compensation from the paper.
    * **Adaptive threshold from optical-flow RMS** when ``adapt=True`` and
      flow data is available.

    When no *sampling_rate* is given the function behaves identically to the
    original implementation.

    Args:
        gaze_data: DataFrame with ``x``, ``y``, ``timestamp`` columns (and
            optionally ``flow_x``, ``flow_y``).
        dispersion_threshold: Maximum dispersion (px) within a fixation window.
        min_fixation_duration: Minimum fixation duration in ms.
        adapt: Enable adaptive threshold.
        tuning_parameter: MAD-based adaptation factor (fallback).
        sampling_rate: Recording sampling rate in Hz.

    Returns:
        Tuple of (DataFrame with events, final threshold).
    """
    (result_data, x, y, t, n, dispersion_threshold, velocity,
     event_type, fixation_x, fixation_y, event_duration, fixation_ids,
     saccade_ids, preprocess_meta) = \
        prepare_classification_data(gaze_data, dispersion_threshold, adapt,
                                    tuning_parameter, is_velocity_based=False,
                                    sampling_rate=sampling_rate)

    # -----------------------------------------------------------------
    # Choose dispersion & coordinate arrays based on preprocessing
    # -----------------------------------------------------------------
    if preprocess_meta is not None:
        disp_col = preprocess_meta["disp_col"]  # "rel_dispersion" or "dispersion"
        disp_values = result_data[disp_col].fillna(0).values
        has_adaptive = preprocess_meta["has_adaptive_threshold"]
        adaptive_thresh = result_data["threshold"].values if has_adaptive else None
        # Use smoothed coordinates for centroid calculation
        cx = result_data["filter_x"].values if "filter_x" in result_data.columns else x
        cy = result_data["filter_y"].values if "filter_y" in result_data.columns else y
        use_precomputed_disp = True
    else:
        use_precomputed_disp = False
        has_adaptive = False
        adaptive_thresh = None
        cx = x
        cy = y

    start_idx = 0
    fixation_id = 1

    while start_idx < n:
        if use_precomputed_disp:
            # -----------------------------------------------------------
            # Enhanced path: use the pre-computed per-sample dispersion.
            # Skip above-threshold samples (saccades) first.
            # -----------------------------------------------------------
            local_thresh_start = adaptive_thresh[start_idx] if has_adaptive else dispersion_threshold
            if disp_values[start_idx] > local_thresh_start:
                start_idx += 1
                continue

            current_idx = start_idx
            while current_idx < n:
                local_thresh = adaptive_thresh[current_idx] if has_adaptive else dispersion_threshold
                if disp_values[current_idx] > local_thresh:
                    break
                current_idx += 1

            # current_idx > start_idx guaranteed (start_idx passed the check)
            end_idx = current_idx - 1
        else:
            # -----------------------------------------------------------
            # Legacy expanding-window dispersion (original behaviour)
            # -----------------------------------------------------------
            current_idx = start_idx
            max_x = x[start_idx]
            min_x = x[start_idx]
            max_y = y[start_idx]
            min_y = y[start_idx]

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

                current_idx += 1

            end_idx = current_idx - 1 if current_idx > start_idx else start_idx

        # Calculate duration of the current window
        window_duration = t[end_idx] - t[start_idx] if end_idx > start_idx else 0.0

        # Check if window meets minimum fixation duration
        if window_duration >= min_fixation_duration:
            event_type[start_idx:end_idx + 1] = 'Fixation'

            fix_x = np.mean(cx[start_idx:end_idx + 1])
            fix_y = np.mean(cy[start_idx:end_idx + 1])

            fixation_x[start_idx:end_idx + 1] = fix_x
            fixation_y[start_idx:end_idx + 1] = fix_y
            event_duration[start_idx:end_idx + 1] = window_duration
            fixation_ids[start_idx:end_idx + 1] = fixation_id
            fixation_id += 1

        start_idx = end_idx + 1 if current_idx > start_idx else start_idx + 1

    saccade_ids = add_saccade_ids(event_type, saccade_ids)

    return finalize_result_dataframe(result_data, event_type, fixation_x, fixation_y, event_duration, fixation_ids, saccade_ids), dispersion_threshold


def classify_ivt(gaze_data, velocity_threshold=0.3, min_fixation_duration=50,
               adapt=False, tuning_parameter=0.1, sampling_rate=None):
    """Identifies fixations using the I-VT (Velocity Threshold) algorithm.

    Enhanced with the I-VAT+Frel pipeline when *sampling_rate* is provided:

    * **Savitzky-Golay smoothing** reduces coordinate noise.
    * **Gaze velocity** is computed on the smoothed coordinates using an
      average time-delta denominator (lower noise).
    * **Flow-relative velocity** (``vel_rel_mag``) replaces raw velocity when
      ``flow_x`` / ``flow_y`` columns are present.  This is the *Frel*
      head-motion compensation.
    * **Adaptive threshold from optical-flow RMS** replaces the MAD-based
      adaptation when flow data is available.

    When no *sampling_rate* is given the function behaves identically to the
    original implementation.

    Args:
        gaze_data: DataFrame with ``x``, ``y``, ``timestamp`` columns (and
            optionally ``flow_x``, ``flow_y``).
        velocity_threshold: Maximum velocity (px/ms) for fixation.
        min_fixation_duration: Minimum fixation duration in ms.
        adapt: Enable adaptive threshold.
        tuning_parameter: MAD-based adaptation factor (fallback).
        sampling_rate: Recording sampling rate in Hz.

    Returns:
        Tuple of (DataFrame with events, final threshold).
    """
    (result_data, x, y, t, n, velocity_threshold, velocity,
     event_type, fixation_x, fixation_y, event_duration, fixation_ids,
     saccade_ids, preprocess_meta) = \
        prepare_classification_data(gaze_data, velocity_threshold, adapt,
                                    tuning_parameter, is_velocity_based=True,
                                    sampling_rate=sampling_rate)

    # -----------------------------------------------------------------
    # Determine whether we have a per-sample adaptive threshold column
    # -----------------------------------------------------------------
    has_adaptive = (preprocess_meta is not None
                    and preprocess_meta.get("has_adaptive_threshold", False))
    adaptive_thresh = result_data["threshold"].values if has_adaptive else None

    # Use smoothed coordinates for centroid calculation when available
    cx = result_data["filter_x"].values if "filter_x" in result_data.columns else x
    cy = result_data["filter_y"].values if "filter_y" in result_data.columns else y

    start_idx = 0
    fixation_id = 1

    while start_idx < n:
        # Skip above-threshold samples (saccades) – they must never be
        # labelled Fixation regardless of min_fixation_duration.
        local_thresh_start = adaptive_thresh[start_idx] if has_adaptive else velocity_threshold
        if velocity[start_idx] > local_thresh_start:
            start_idx += 1
            continue

        # Find consecutive points with velocity under threshold
        current_idx = start_idx
        while current_idx < n:
            local_thresh = adaptive_thresh[current_idx] if has_adaptive else velocity_threshold
            if velocity[current_idx] > local_thresh:
                break
            current_idx += 1

        # current_idx > start_idx is guaranteed here (start_idx passed the
        # threshold check above, so the inner loop advanced at least once).
        end_idx = current_idx - 1
        window_duration = t[end_idx] - t[start_idx] if end_idx > start_idx else 0.0

        # Validate potential fixation duration
        if window_duration >= min_fixation_duration:
            event_type[start_idx:end_idx + 1] = 'Fixation'

            fix_x = np.mean(cx[start_idx:end_idx + 1])
            fix_y = np.mean(cy[start_idx:end_idx + 1])

            fixation_x[start_idx:end_idx + 1] = fix_x
            fixation_y[start_idx:end_idx + 1] = fix_y
            event_duration[start_idx:end_idx + 1] = window_duration
            fixation_ids[start_idx:end_idx + 1] = fixation_id
            fixation_id += 1

        # Advance to next potential fixation window
        start_idx = end_idx + 1

    saccade_ids = add_saccade_ids(event_type, saccade_ids)

    return finalize_result_dataframe(result_data, event_type, fixation_x, fixation_y,
                                     event_duration, fixation_ids, saccade_ids), velocity_threshold
