
import pandas as pd
import numpy as np
import logging
from sklearn.metrics import calinski_harabasz_score
from .utils import clean_fixations, merge_close_fixations
from .detection_algorithms import classify_idt, classify_ivt


def optimize_threshold(gaze_data, min_fixation_duration=50, adapt=False, candidate_thresholds=None):
    """
    Optimize dispersion threshold using cluster validity metrics (Calinski-Harabasz score).

    Args:
        gaze_data (pd.DataFrame): DataFrame with ['x', 'y', 'timestamp'] columns.
        min_fixation_duration (float): Minimum fixation duration in ms.
        adapt (bool): Whether to adapt the dispersion threshold based on velocity.
        candidate_thresholds (list, optional): List of dispersion thresholds to test.

    Returns:
        float: Best threshold according to Calinski-Harabasz score.
    """
    # TODO: Ask: use real candidate thresholds based on data characteristics?
    if candidate_thresholds is None:
        candidate_thresholds = [125, 150, 175, 200, 225, 250, 275, 300]

    best_score = -np.inf
    best_threshold = candidate_thresholds[0]

    for threshold in candidate_thresholds:
        try:
            # TODO: I-VT implementation
            # Run I-DT classification
            classified_df = classify_idt(
                gaze_data.copy(),
                dispersion_threshold=threshold,
                min_fixation_duration=min_fixation_duration,
                adapt=adapt
            )

            # Extract valid fixation points
            fixation_mask = classified_df['event_type'] == 'Fixation'
            fixation_points = classified_df[fixation_mask][['fixation_x', 'fixation_y']].values

            # Get cluster labels (fixation IDs)
            labels = classified_df[fixation_mask]['fixation_id'].values

            # Need at least 2 fixations to compute CH score
            if len(np.unique(labels)) < 2:
                continue

            # Compute Calinski-Harabasz score
            score = calinski_harabasz_score(fixation_points, labels)

            if score > best_score:
                best_score = score
                best_threshold = threshold

        except Exception as e:
            print(f"Error with threshold {threshold}: {str(e)}")
            continue

    return best_threshold


def classify_aoi(self, gaze_data, aois, algorithm='weighted_bbox_attach'):
    """
    Checks if the fixation points are in the AOI or not.

    Args:
        gaze_data (pd.DataFrame): DataFrame containing gaze data with fixation coordinates.
        aois (pd.DataFrame): DataFrame containing AOI definitions.
        algorithm (str): Algorithm to use for AOI classification. Options:
            - 'standard': Checks if fixation is within AOI boundaries.
            - 'attach': Attaches each fixation to its closest AOI.
            - 'bbox_attach': Attaches to closest AOI bounding box.
            - 'weighted_bbox_attach'-default: Attaches to closest AOI with weighting.

    Returns:
        pd.DataFrame: Gaze data with added AOI classification columns.
    """
    if not self.is_valid_data:
        return None, None

    # first get all unique fixation points
    # Create a df with unique fixation points
    fixations = gaze_data[['fixation_id', 'fixation_x', 'fixation_y']].drop_duplicates()

    # remove NaN values
    fixations = fixations[fixations['fixation_x'].notna() & fixations['fixation_y'].notna()]

    # Add columns for AOI classification
    fixations['aoi_type'] = np.nan
    fixations['aoi'] = np.nan

    # reset index
    fixations.reset_index(drop=True, inplace=True)

    if algorithm == 'standard':
        # Standard algorithm: check if fixation is within AOI boundaries
        for index, row in fixations.iterrows():
            for aoi_index, aoi in aois.iterrows():
                # aois are in format ['aoi_type', 'aoi', 'pos_x', 'pos_y', 'width', 'height']
                # Each aoi has a bounding box defined by its top-left corner ('pos_x', 'pos_y') and its size ('width', 'height')
                if aoi['pos_x'] <= row['fixation_x'] <= aoi['pos_x'] + aoi['width'] \
                        and aoi['pos_y'] <= row['fixation_y'] <= aoi['pos_y'] + aoi['height']:
                    fixations.loc[index, 'aoi_type'] = aoi['aoi_type']
                    fixations.loc[index, 'aoi'] = aoi['aoi']
                    fixations.loc[index, 'aoi_id'] = aoi_index
                    break

    elif algorithm == 'attach':
        # Attach algorithm: attach each fixation to its closest AOI centroid
        for index, row in fixations.iterrows():
            min_distance = float('inf')
            closest_aoi = None
            closest_aoi_index = None

            for aoi_index, aoi in aois.iterrows():
                # Calculate the center of the AOI
                aoi_center_x = aoi['pos_x'] + aoi['width'] / 2
                aoi_center_y = aoi['pos_y'] + aoi['height'] / 2

                # Calculate Euclidean distance from fixation to AOI center
                distance = np.sqrt((row['fixation_x'] - aoi_center_x) ** 2 +
                                    (row['fixation_y'] - aoi_center_y) ** 2)

                # Update closest AOI if this one is closer
                if distance < min_distance:
                    min_distance = distance
                    closest_aoi = aoi
                    closest_aoi_index = aoi_index

            # Assign the closest AOI to this fixation
            if closest_aoi is not None:
                fixations.loc[index, 'aoi_type'] = closest_aoi['aoi_type']
                fixations.loc[index, 'aoi'] = closest_aoi['aoi']
                fixations.loc[index, 'aoi_id'] = closest_aoi_index

    elif algorithm == 'bbox_attach':
        # Bbox Attach algorithm: attach each fixation to the AOI with the closest bounding box
        for index, row in fixations.iterrows():
            min_distance = float('inf')
            closest_aoi = None
            closest_aoi_index = None

            for aoi_index, aoi in aois.iterrows():
                # Calculate distance to the nearest point on the AOI's bounding box
                # First, determine the nearest point on the bbox
                x = row['fixation_x']
                y = row['fixation_y']

                # Bounding box coordinates
                left = aoi['pos_x']
                right = aoi['pos_x'] + aoi['width']
                top = aoi['pos_y']
                bottom = aoi['pos_y'] + aoi['height']

                # Find closest x-coordinate on bbox
                if x < left:
                    nearest_x = left
                elif x > right:
                    nearest_x = right
                else:
                    nearest_x = x  # x is within the bbox horizontally

                # Find closest y-coordinate on bbox
                if y < top:
                    nearest_y = top
                elif y > bottom:
                    nearest_y = bottom
                else:
                    nearest_y = y  # y is within the bbox vertically

                # Calculate distance to nearest point on bbox
                distance = np.sqrt((x - nearest_x) ** 2 + (y - nearest_y) ** 2)

                # Update closest AOI if this one is closer
                if distance < min_distance:
                    min_distance = distance
                    closest_aoi = aoi
                    closest_aoi_index = aoi_index

            # Assign the closest AOI to this fixation
            if closest_aoi is not None:
                fixations.loc[index, 'aoi_type'] = closest_aoi['aoi_type']
                fixations.loc[index, 'aoi'] = closest_aoi['aoi']
                fixations.loc[index, 'aoi_id'] = closest_aoi_index

    elif algorithm == 'weighted_bbox_attach':
        # Weighted Bbox Attach algorithm: attach each fixation to the AOI with the closest weighted distance
        # This allows you to prioritize certain AOIs (like text) over others (like images) when a fixation is near multiple AOIs.
        for index, row in fixations.iterrows():
            min_weighted_distance = float('inf')
            closest_aoi = None
            closest_aoi_index = None

            for aoi_index, aoi in aois.iterrows():
                # Calculate distance to the nearest point on the AOI's bounding box
                x = row['fixation_x']
                y = row['fixation_y']

                # Bounding box coordinates
                left = aoi['pos_x']
                right = aoi['pos_x'] + aoi['width']
                top = aoi['pos_y']
                bottom = aoi['pos_y'] + aoi['height']

                # Find closest x-coordinate on bbox
                if x < left:
                    nearest_x = left
                elif x > right:
                    nearest_x = right
                else:
                    nearest_x = x  # x is within the bbox horizontally

                # Find closest y-coordinate on bbox
                if y < top:
                    nearest_y = top
                elif y > bottom:
                    nearest_y = bottom
                else:
                    nearest_y = y  # y is within the bbox vertically

                # Calculate distance to nearest point on bbox
                distance = np.sqrt((x - nearest_x) ** 2 + (y - nearest_y) ** 2)

                # Apply weighting based on AOI type
                weight = 1.0
                if aoi['aoi_type'] == 'word' in str(aoi['aoi']).lower():
                    # Give text/caption AOIs higher priority (lower weighted distance)
                    weight = 0.5
                elif 'image' in str(aoi['aoi_type']).lower():
                    # Give image AOIs lower priority (higher weighted distance)
                    weight = 3

                # Calculate distance after applying the weight
                weighted_distance = distance * weight

                # Update closest AOI if this one is closer after weighting
                if weighted_distance < min_weighted_distance:
                    min_weighted_distance = weighted_distance
                    closest_aoi = aoi
                    closest_aoi_index = aoi_index

            # Assign the closest AOI to this fixation
            if closest_aoi is not None:
                fixations.loc[index, 'aoi_type'] = closest_aoi['aoi_type']
                fixations.loc[index, 'aoi'] = closest_aoi['aoi']
                fixations.loc[index, 'aoi_id'] = closest_aoi_index

    else:
        raise ValueError(f"Unsupported AOI classification algorithm: {algorithm}")

    # merge the aoi column with the gaze_data on fixation_id
    gaze_data = pd.merge(gaze_data, fixations[['fixation_id', 'aoi_type', 'aoi', 'aoi_id']], on='fixation_id', how='left')

    return gaze_data


def detect_event(self, plot=False, min_fixation_duration=50.0, aois=None, algorithm=None,
                threshold=150.0, adapt=False, optimize=False):
    """Detects events in the loaded gaze data using the specified algorithm.

    Args:
        plot (bool): Whether to plot the event segmentation.
        min_fixation_duration (float): Minimum duration for a fixation (in milliseconds).
        aois (pd.DataFrame): Areas of interest for AOI classification.
        algorithm (str): Event detection algorithm ('idt'-default or 'ivt').
        threshold (float): Threshold (for 'idt' or 'ivt' algorithm, in pixels).
        adapt (bool): Whether to adapt the threshold based on velocity.
        optimize (bool): Whether to optimize the dispersion threshold.

    Returns:
        pd.DataFrame: Processed gaze data with event classifications.

    """
    if not self.is_valid_data:
        return None
    
    # Find the best dispersion threshold using optimization
    best_thresh = optimize_threshold(self.gaze_data, adapt=adapt) if optimize else threshold
    print(f"Best Threshold:{best_thresh}")

    try:
        if algorithm == 'idt':
            # Run I-DT with the best threshold
            data = classify_idt(self.gaze_data, dispersion_threshold=best_thresh,
                                        min_fixation_duration=min_fixation_duration, adapt=adapt)
        elif algorithm == 'ivt':
                data = classify_ivt(self.gaze_data, velocity_threshold=best_thresh,
                                        min_fixation_duration=min_fixation_duration, adapt=adapt)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

    except Exception as e:
        logging.error(f"Error detecting events: {e}")
        return None

    # Assign each fixation to an AOI if AOIs are provided
    if aois is not None:
        data = classify_aoi(data, aois)

    return data


def detect_event_with_merge(self, plot=False, min_fixation_duration=50.0, aois=None,
                            algorithm=None, threshold=150.0, merge_distance=100.0, adapt=False,
                            optimize=False, return_threshold=False):
    """
    Detects events in the loaded gaze data using the specified algorithm and merges close fixations.

    Args:
        plot (bool): Whether to plot the event segmentation.
        min_fixation_duration (float): Minimum duration for a fixation (in milliseconds).
        aois (pd.DataFrame): Areas of interest for AOI classification.
        algorithm (str): Event detection algorithm ('ivt' or 'idt'-default).
        threshold (float): Threshold (for 'idt' or 'ivt' algorithm, in pixels).
        merge_distance (float): Maximum distance for merging consecutive fixations.
        adapt (bool): Whether to adapt the dispersion threshold based on velocity.
        optimize (bool): Whether to optimize the dispersion threshold.
        return_threshold (bool): Whether to return the optimized threshold.

    Returns:
        pd.DataFrame: Processed gaze data with event classifications and merged fixations.
    """
    if not self.is_valid_data:
        return None
    
    # Find the best dispersion threshold using optimization
    best_thresh = optimize_threshold(self.gaze_data, adapt=adapt) if optimize else threshold
    print(f"Best Threshold:{best_thresh}")

    try:
        # First detect events using the original algorithm
        if algorithm == 'idt':
            data = classify_idt(self.gaze_data, dispersion_threshold=best_thresh,
                                    min_fixation_duration=min_fixation_duration, adapt=adapt)
        elif algorithm == 'ivt':
            data = classify_ivt(self.gaze_data, velocity_threshold=best_thresh,
                                    min_fixation_duration=min_fixation_duration, adapt=adapt)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        # Merge close fixations
        data = merge_close_fixations(data, distance_threshold=merge_distance)

        if aois is not None:
            data = classify_aoi(data, aois)

    except Exception as e:
        logging.error(f"Error detecting events: {e}")
        return None

    if return_threshold:
        return data, best_thresh

    return data


def process_event(self, output_dir, plot=True,
                            min_fixation_duration=50.0, aoi_file_path=None, algorithm=None,
                            merge_distance: float = None, threshold=150.0, adapt=False,
                            optimize=False, duration_cutoff: float = None):
    """
    Processes event detection for a single user and image with fixation merging and stores the output.

    Args:
        output_dir (str): Directory to store the output CSV files.
        plot (bool): Whether to plot the event segmentation.
        min_fixation_duration (float): Minimum duration for fixation events.
        aoi_file_path (str): Whether to check if the fixation points are in the AOI or not
        algorithm: Event detection algorithm ('ivt' or 'idt').
        merge_distance (float): Maximum distance for merging consecutive fixations.
        threshold (float): Threshold for the I-DT or I-VTS algorithm.
        adapt (bool): Whether to adapt the dispersion threshold based on velocity.
        optimize (bool): Whether to optimize the dispersion threshold.
        duration_cutoff (float): Optional duration cutoff.

    Returns:
        bool: True if processing was successful, False otherwise.
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

    if merge_distance:
        event_gaze = detect_event_with_merge(
            self,
            plot=plot,
            min_fixation_duration=min_fixation_duration,
            aois=aois,
            algorithm=algorithm,
            merge_distance=merge_distance,
            threshold=threshold,
            adapt=adapt,
            optimize=optimize
        )
    else:
        event_gaze = detect_event(
            self,
            plot=plot,
            min_fixation_duration=min_fixation_duration,
            aois=aois,
            algorithm=algorithm,
            threshold=threshold,
        )

    if event_gaze is not None:
        event_gaze = clean_fixations(event_gaze)
        event_gaze.to_csv(output_dir, index=False, sep=';')
        logging.info(f"Processed and saved event data in {output_dir}")
        return True
    else:
        logging.error("Failed to process event data")
        return False
