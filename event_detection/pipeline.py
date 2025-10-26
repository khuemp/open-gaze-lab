import pandas as pd
import numpy as np
import logging
from sklearn.metrics import calinski_harabasz_score
from .utils import clean_fixations, merge_fixations, merge_saccades
from .detection_algorithms import classify_idt, classify_ivt


def classify_aoi(self, gaze_data, aois, algorithm='weighted_bbox_attach'):
    """Classifies fixation points based on their relationship to Areas of Interest (AOIs).

    Determines which AOI, if any, each fixation point belongs to using various classification
    algorithms. Can use standard containment, closest AOI attachment, or weighted bounding box
    methods for classification.

    Args:
        gaze_data (pd.DataFrame): Fixation data with columns:
            - fixation_id (int): Unique fixation identifier
            - fixation_x (float): X coordinate of fixation
            - fixation_y (float): Y coordinate of fixation
        aois (pd.DataFrame): AOI definitions with columns:
            - aoi_type (str): Category or type of the AOI
            - aoi (str): Name or identifier of the AOI
            - pos_x (float): X coordinate of AOI's top-left corner
            - pos_y (float): Y coordinate of AOI's top-left corner
            - width (float): Width of the AOI
            - height (float): Height of the AOI
        algorithm (str, optional): AOI classification method to use:
            - 'standard': Point containment within AOI bounds
            - 'attach': Assign to nearest AOI center
            - 'bbox_attach': Use expanded bounding boxes
            - 'weighted_bbox_attach': Distance-weighted assignment
            Defaults to 'weighted_bbox_attach'.

    Returns:
        pd.DataFrame: Input data with added columns:
            - aoi_type (str): Type of the assigned AOI
            - aoi (str): Name of the assigned AOI
            - aoi_id (int): Index of the assigned AOI
    """
    if not hasattr(self, 'is_valid_data') or not self.is_valid_data:
        # Proceed even without class context
        pass

    # Extract unique fixation points from gaze data
    fixations = gaze_data[['fixation_id', 'fixation_x', 'fixation_y']].drop_duplicates()

    # Filter out invalid coordinates
    fixations = fixations[fixations['fixation_x'].notna() & fixations['fixation_y'].notna()]

    # Initialize AOI classification columns
    fixations['aoi_type'] = np.nan
    fixations['aoi'] = np.nan
    fixations['aoi_id'] = np.nan

    # Prepare for processing
    fixations.reset_index(drop=True, inplace=True)

    if algorithm == 'standard':
        for index, row in fixations.iterrows():
            for aoi_index, aoi in aois.iterrows():
                if aoi['pos_x'] <= row['fixation_x'] <= aoi['pos_x'] + aoi['width'] and \
                   aoi['pos_y'] <= row['fixation_y'] <= aoi['pos_y'] + aoi['height']:
                    fixations.loc[index, 'aoi_type'] = aoi['aoi_type']
                    fixations.loc[index, 'aoi'] = aoi['aoi']
                    fixations.loc[index, 'aoi_id'] = aoi_index
                    break
    elif algorithm == 'attach':
        for index, row in fixations.iterrows():
            min_distance = float('inf')
            closest_aoi = None
            closest_aoi_index = None
            for aoi_index, aoi in aois.iterrows():
                aoi_center_x = aoi['pos_x'] + aoi['width'] / 2
                aoi_center_y = aoi['pos_y'] + aoi['height'] / 2
                distance = np.sqrt((row['fixation_x'] - aoi_center_x)**2 + (row['fixation_y'] - aoi_center_y)**2)
                if distance < min_distance:
                    min_distance = distance
                    closest_aoi = aoi
                    closest_aoi_index = aoi_index
            if closest_aoi is not None:
                fixations.loc[index, 'aoi_type'] = closest_aoi['aoi_type']
                fixations.loc[index, 'aoi'] = closest_aoi['aoi']
                fixations.loc[index, 'aoi_id'] = closest_aoi_index
    # Additional algorithms can be implemented using similar coordinate-based logic
    else:
        raise ValueError(f"Unsupported AOI classification algorithm: {algorithm}")

    # Merge classification results back into original data
    gaze_data = pd.merge(gaze_data, fixations[['fixation_id', 'aoi_type', 'aoi', 'aoi_id']], on='fixation_id', how='left')
    return gaze_data


def optimize_threshold(gaze_data, min_fixation_duration=50, adapt=False, algorithm=None, candidate_thresholds=None):
    """Optimizes detection threshold using cluster validity metrics.

    Uses the Calinski-Harabasz score to evaluate the quality of fixation clusters produced
    by different threshold values. Iteratively tests multiple threshold values to find the
    one that produces the most well-defined fixation clusters. The optimal threshold
    maximizes between-cluster separation while minimizing within-cluster spread.

    Args:
        gaze_data (pd.DataFrame): Input gaze data with columns:
            - x (float): X coordinate in pixels
            - y (float): Y coordinate in pixels
            - timestamp (float): Time in milliseconds
        min_fixation_duration (int, optional): Minimum time in milliseconds for a
            fixation to be valid. Defaults to 50ms.
        adapt (bool, optional): Whether to use adaptive thresholding based on
            data characteristics. Defaults to False.
        algorithm (str, optional): Detection algorithm to use. Must be either
            'idt' (dispersion-based) or 'ivt' (velocity-based). Defaults to None.
        candidate_thresholds (list[float], optional): List of threshold values to
            evaluate. If None, uses [125, 150, 175, 200, 225, 250, 275, 300].

    Returns:
        float: Best performing threshold value based on cluster validity score.
            Returns the first threshold if optimization fails.

    Notes:
        - Uses scikit-learn's Calinski-Harabasz score for cluster evaluation
        - Requires at least 2 fixations to compute cluster validity
        - Silently skips thresholds that produce invalid clustering results
        - Higher scores indicate better-defined, more distinct fixation clusters
    """
    if candidate_thresholds is None:
        candidate_thresholds = [125, 150, 175, 200, 225, 250, 275, 300]

    best_score = -np.inf
    best_threshold = candidate_thresholds[0]

    for detect_threshold in candidate_thresholds:
        try:
            if algorithm == 'idt':
                classified_df = classify_idt(gaze_data.copy(), dispersion_threshold=detect_threshold, min_fixation_duration=min_fixation_duration, adapt=adapt)
            else:
                classified_df = classify_ivt(gaze_data.copy(), velocity_threshold=detect_threshold, min_fixation_duration=min_fixation_duration, adapt=adapt)

            fixation_mask = classified_df['event_type'] == 'Fixation'
            fixation_points = classified_df[fixation_mask][['fixation_x', 'fixation_y']].values
            labels = classified_df[fixation_mask]['fixation_id'].values

            if len(np.unique(labels)) < 2:
                continue

            score = calinski_harabasz_score(fixation_points, labels)

            if score > best_score:
                best_score = score
                best_threshold = detect_threshold
        except Exception as e:
            print(f"Error with threshold {detect_threshold}: {str(e)}")
            continue

    return best_threshold


def detect_event(self, min_fixation_duration=50, aois=None,
                            algorithm=None, detect_threshold=150.0, fixation_merge_threshold=None, adapt=False,
                            tuning_parameter=0.1, optimize=False):
    """Detects fixation and saccade events in gaze data using specified algorithm.

    Processes raw gaze data to identify fixations and saccades, optionally optimizes
    detection parameters, merges events if specified, and classifies fixations into AOIs.
    Uses either I-DT or I-VT algorithms for event detection.

    Args:
        min_fixation_duration (int, optional): Minimum time in milliseconds for a
            fixation to be valid. Defaults to 50ms.
        aois (pd.DataFrame, optional): Areas of Interest definitions. Should contain
            columns: ['aoi_type', 'aoi', 'pos_x', 'pos_y', 'width', 'height'].
            Defaults to None.
        algorithm (str, optional): Detection algorithm to use ('idt' or 'ivt').
            Defaults to None.
        detect_threshold (float, optional): Initial threshold for detection algorithm.
            For IVT: velocity in pixels/ms. For IDT: dispersion in pixels.
            Defaults to 150.0.
        fixation_merge_threshold (float, optional): Maximum distance in pixels between
            fixations to be merged. If None, no merging occurs. Defaults to None.
        adapt (bool, optional): Whether to use adaptive thresholding based on
            data characteristics. Defaults to False.
        tuning_parameter (float, optional): Sensitivity factor for adaptive
            threshold adjustment. Defaults to 0.1.
        optimize (bool, optional): Whether to optimize detection threshold using
            cluster validity metrics. Defaults to False.

    Returns:
        tuple:
            - pd.DataFrame or None: Processed gaze data with detected events and
              their properties. None if processing fails.
            - float or None: Final detection threshold used. If optimize=True,
              this is the optimized threshold. None if processing fails.

    Notes:
        - Requires valid gaze data to be present in self.gaze_data
        - Returns None, None if self.is_valid_data is False
        - Event detection can fail if data format is incorrect or contains invalid values
    """
    if not self.is_valid_data:
        return None, None
    
    best_thresh = optimize_threshold(self.gaze_data, adapt=adapt, algorithm=algorithm) if optimize else detect_threshold
    print(f"Best Threshold: {best_thresh}")

    try:
        if algorithm == 'idt':
            data = classify_idt(self.gaze_data, dispersion_threshold=best_thresh, min_fixation_duration=min_fixation_duration, adapt=adapt, tuning_parameter=tuning_parameter)
        elif algorithm == 'ivt':
            data = classify_ivt(self.gaze_data, velocity_threshold=best_thresh, min_fixation_duration=min_fixation_duration, adapt=adapt, tuning_parameter=tuning_parameter)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        if fixation_merge_threshold is not None:
            data = merge_fixations(data, fixation_merge_threshold=fixation_merge_threshold)
        
        data = merge_saccades(data)

        if aois is not None:
            data = classify_aoi(self, data, aois)

    except Exception as e:
        logging.error(f"Error detecting events: {e}")
        return None, None

    return data, best_thresh


def process_event(self, output_dir, min_fixation_duration=50, aoi_file_path=None, algorithm=None,
                            fixation_merge_threshold: float = None, detect_threshold=150.0, adapt=False,
                            tuning_parameter=0.1, optimize=False, duration_cutoff: float = None):
    """Processes and saves eye-tracking event detection results to a CSV file.

    Performs complete event detection pipeline including optional duration trimming,
    fixation/saccade detection, AOI classification, and data cleaning. Saves results
    as a detailed CSV with one row per gaze point.

    Args:
        output_dir (str): Path where the output CSV file will be saved.
        min_fixation_duration (int, optional): Minimum time in milliseconds for a
            fixation to be valid. Defaults to 50ms.
        aoi_file_path (str, optional): Path to CSV file containing AOI definitions.
            Should have columns: ['aoi_type', 'aoi', 'pos_x', 'pos_y', 'width', 'height'].
            Defaults to None.
        algorithm (str, optional): Detection algorithm to use ('idt' or 'ivt').
            Defaults to None.
        fixation_merge_threshold (float, optional): Maximum distance in pixels between
            fixations to be merged. If None, no merging occurs. Defaults to None.
        detect_threshold (float, optional): Initial threshold for detection algorithm.
            For IVT: velocity in pixels/ms. For IDT: dispersion in pixels.
            Defaults to 150.0.
        adapt (bool, optional): Whether to use adaptive thresholding based on
            data characteristics. Defaults to False.
        tuning_parameter (float, optional): Sensitivity factor for adaptive
            threshold adjustment. Defaults to 0.1.
        optimize (bool, optional): Whether to optimize detection threshold using
            cluster validity metrics. Defaults to False.
        duration_cutoff (float, optional): Maximum duration in milliseconds to process
            from the end of the recording. If set, only processes the last X milliseconds.
            Defaults to None (process all data).

    Returns:
        pd.DataFrame or None: Processed gaze data with all detected events and
            their properties. Returns None if processing fails.

    Notes:
        - Output CSV uses semicolon (;) as delimiter
        - Timestamps are adjusted to start at 0 when duration_cutoff is used
        - All fixations are cleaned and recalculated before saving
        - Logs success/failure messages through logging module
    """

    aois = None
    if aoi_file_path is not None:
        aois = pd.read_csv(aoi_file_path)

    if duration_cutoff is not None:
        total_duration = self.gaze_data["timestamp"].iloc[-1]
        if total_duration > duration_cutoff:
            start_threshold = total_duration - duration_cutoff
            self.gaze_data = self.gaze_data.loc[self.gaze_data["timestamp"] >= start_threshold].reset_index(drop=True)
            t0 = self.gaze_data["timestamp"].iloc[0]
            self.gaze_data["timestamp"] = self.gaze_data["timestamp"] - t0
            
    event_gaze, best_thresh = detect_event(
        self,
        min_fixation_duration=min_fixation_duration,
        aois=aois,
        algorithm=algorithm,
        fixation_merge_threshold=fixation_merge_threshold,
        detect_threshold=detect_threshold,
        adapt=adapt,
        tuning_parameter=tuning_parameter,
        optimize=optimize
    )

    if event_gaze is not None:
        # Process fixations to ensure consistent duration calculations
        final_gaze_data = clean_fixations(event_gaze)
        
        final_gaze_data.to_csv(output_dir, index=False, sep=';')
        logging.info(f"Processed and saved event data in {output_dir}")
    else:
        logging.error("Failed to process event data")
        return None
    
    return final_gaze_data

