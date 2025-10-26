import pandas as pd
import numpy as np
import logging
from sklearn.metrics import calinski_harabasz_score
from .utils import clean_fixations, merge_fixations, merge_saccades
from .detection_algorithms import classify_idt, classify_ivt


def classify_aoi(self, gaze_data, aois, algorithm='weighted_bbox_attach'):
    """
    Checks if the fixation points are in the AOI or not.
    (Note: Corrected to use 'fixation_x' and 'fixation_y')
    """
    if not hasattr(self, 'is_valid_data') or not self.is_valid_data:
        # Check if running in a class context, otherwise proceed
        pass

    # Create a df with unique fixation points
    fixations = gaze_data[['fixation_id', 'fixation_x', 'fixation_y']].drop_duplicates()

    # remove NaN values
    fixations = fixations[fixations['fixation_x'].notna() & fixations['fixation_y'].notna()]

    # Add columns for AOI classification
    fixations['aoi_type'] = np.nan
    fixations['aoi'] = np.nan
    fixations['aoi_id'] = np.nan

    # reset index
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
    # ... (other algorithms like bbox_attach would also use fixation_x/y) ...
    else:
        raise ValueError(f"Unsupported AOI classification algorithm: {algorithm}")

    # merge the aoi column with the gaze_data on fixation_id
    gaze_data = pd.merge(gaze_data, fixations[['fixation_id', 'aoi_type', 'aoi', 'aoi_id']], on='fixation_id', how='left')
    return gaze_data


def optimize_threshold(gaze_data, min_fixation_duration=50, adapt=False, algorithm=None, candidate_thresholds=None):
    """
    Optimize dispersion threshold using cluster validity metrics (Calinski-Harabasz score).
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
                            optimize=False):
    """
    Detects events, merges them if a threshold is provided, and re-indexes saccades.
    """
    if not self.is_valid_data:
        return None, None
    
    best_thresh = optimize_threshold(self.gaze_data, adapt=adapt, algorithm=algorithm) if optimize else detect_threshold
    print(f"Best Threshold: {best_thresh}")

    try:
        if algorithm == 'idt':
            data = classify_idt(self.gaze_data, dispersion_threshold=best_thresh, min_fixation_duration=min_fixation_duration, adapt=adapt)
        elif algorithm == 'ivt':
            data = classify_ivt(self.gaze_data, velocity_threshold=best_thresh, min_fixation_duration=min_fixation_duration, adapt=adapt)
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
                            optimize=False, duration_cutoff: float = None):
    """
    Processes event detection and saves the result as a detailed, 
    one-row-per-gaze-point CSV file for all scenarios.
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
        optimize=optimize
    )

    if event_gaze is not None:
        # ALWAYS use clean_fixations to get the detailed, one-row-per-gaze-point output.
        # This function will correctly calculate durations for both merged and unmerged data.
        final_gaze_data = clean_fixations(event_gaze)
        
        final_gaze_data.to_csv(output_dir, index=False, sep=';')
        logging.info(f"Processed and saved event data in {output_dir}")
    else:
        logging.error("Failed to process event data")
        return None
    
    return final_gaze_data

