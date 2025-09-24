import os
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as pyo
import logging
from sklearn.metrics import calinski_harabasz_score

class EventDetection:
    """
    Class for detecting events in gaze data.

    Args:
        dataset_loader (DatasetLoaderV2): DatasetLoaderV2 instance to load gaze data.

    Methods:
        load_gaze_data(user, image): Loads gaze data for the specified user and image.
        detect_event(detect_plot=False): Detects events in the loaded gaze data and returns a tuple of discrete time intervals and class labels.
    """

    def __init__(self, loaded_gaze_df):
        """Initializes the EventDetection class."""
        self.gaze_data = loaded_gaze_df.copy()
        self.gaze_data = self.gaze_data.loc[~((self.gaze_data["x"] == 0.0) & (self.gaze_data["y"] == 1.0))].reset_index(drop=True)
        self.is_valid_data = True

        self.gaze_data["y"] = 1 - self.gaze_data["y"]  # Invert y-axis to match image coordinates

        # Configure logger
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
        )

    def compute_velocity(self, df):
        """
        Compute velocity for variable framerate data (5-30fps).
        No modifications to original timestamps or data.
        """
        # Ensure no NaNs in x, y, or timestamp
        df_clean = df.dropna(subset=['x', 'y', 'timestamp'])
        
        if len(df_clean) < 2:
            return np.array([])
        
        # Compute differences
        dx = df_clean['x'].diff().values[1:]  # Remove first NaN from diff()
        dy = df_clean['y'].diff().values[1:]  # Remove first NaN from diff()
        dt = df_clean['timestamp'].diff().values[1:]  # Remove first NaN from diff()
        
        # For variable framerate (5-30fps), minimum time difference should be ~0.033s (30fps)
        # Use this as epsilon to avoid division by zero
        min_dt = 0.033  # 30fps as minimum reasonable interval
        dt[dt <= 0] = min_dt
        
        # Compute velocity (coordinates per second)
        velocity = np.sqrt(dx ** 2 + dy ** 2) / dt
        
        return velocity

    def compute_mad(self, velocity):
        """
        Compute Median Absolute Deviation with robust handling.
        
        The Median Absolute Deviation (MAD) is a measure of data variability
        Adaptive Threshold Logic
        - High MAD: Large velocity variability → More noise → Increase dispersion threshold.
        - Low MAD: Stable velocities → Less noise → Use smaller dispersion threshold.
        
        :param velocity:
        :return:
        """
        # Remove invalid values
        valid_velocity = velocity[~np.isnan(velocity) & np.isfinite(velocity)]
        
        if len(valid_velocity) == 0:
            return 0.0
        
        # Compute median velocity
        median_vel = np.median(valid_velocity)
        
        # Compute absolute deviations from the median
        deviations = np.abs(valid_velocity - median_vel)
        
        # Compute MAD
        mad = np.median(deviations)
        
        return mad

    def classify_idt(self, gaze_data, dispersion_threshold=100.0, min_fixation_duration=50.0, adapt_velocity=False):
        """
        Classifies gaze points into fixations and saccades using the I-DT algorithm.
        Works with variable framerate data without modifying original timestamps.
        
        Args:
            gaze_data (pd.DataFrame): DataFrame containing gaze data with 'x', 'y', and 'timestamp' columns.
            dispersion_threshold (float): Maximum allowed dispersion within a fixation window (in pixels).
            min_fixation_duration (float): Minimum duration (in milliseconds) for a fixation to be considered valid.
            adapt_velocity (bool): Whether to adapt the dispersion threshold based on velocity.

        Returns:
            pd.DataFrame: Gaze data with added 'event_type', 'fixation_x', 'fixation_y', and 'event_duration' columns.
        """
        # Work with a copy to avoid modifying original data
        result_data = gaze_data.copy()
        
        # Scale coordinates to pixels for dispersion calculation
        x = result_data['x'].values * 640
        y = result_data['y'].values * 480
        t = result_data['timestamp'].values

        n = len(x)
        event_type = np.full(n, 'Saccade', dtype='U10')
        fixation_x = np.full(n, np.nan)
        fixation_y = np.full(n, np.nan)
        event_duration = np.full(n, np.nan)
        fixation_ids = np.zeros(n)

        start_idx = 0
        fixation_id = 1

        # Adaptive threshold computation
        original_threshold = dispersion_threshold
        if adapt_velocity:
            # Create temporary dataframe with scaled coordinates for velocity computation
            temp_df = pd.DataFrame({
                'x': x,
                'y': y, 
                'timestamp': t
            })
            
            velocity = self.compute_velocity(temp_df)
            if len(velocity) > 0:
                mad_velocity = self.compute_mad(velocity)
                
                if mad_velocity > 0:
                    # Adaptive threshold: higher MAD = more noise = higher threshold
                    alpha = 0.1  # Tuning parameter
                    adaptation_factor = 1 + alpha * mad_velocity
                    dispersion_threshold = original_threshold * adaptation_factor
                    
                    print(f"Original threshold: {original_threshold:.2f}, "
                          f"MAD velocity: {mad_velocity:.4f}, "
                          f"Adaptive threshold: {dispersion_threshold:.2f}")
                else:
                    print(f"Using original threshold: {dispersion_threshold} (MAD = 0)")
            else:
                print(f"Using original threshold: {dispersion_threshold} (no valid velocity)")

        # Convert min_fixation_duration from milliseconds to seconds for comparison with timestamps
        min_duration_seconds = min_fixation_duration / 1000.0

        while start_idx < n:
            current_idx = start_idx
            max_x = x[start_idx]
            min_x = x[start_idx]
            max_y = y[start_idx]
            min_y = y[start_idx]

            # Expand window until dispersion exceeds threshold
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

            # Calculate duration of the current window
            end_idx = current_idx - 1 if current_idx > start_idx else start_idx
            window_duration_seconds = t[end_idx] - t[start_idx] if end_idx > start_idx else 0.0

            # Check if window meets minimum fixation duration
            if window_duration_seconds >= min_duration_seconds:
                # Classify as fixation
                event_type[start_idx:end_idx + 1] = 'Fixation'
                
                # Calculate fixation center in normalized coordinates
                fix_x = np.mean(x[start_idx:end_idx + 1]) / 640
                fix_y = np.mean(y[start_idx:end_idx + 1]) / 480
                
                fixation_x[start_idx:end_idx + 1] = fix_x
                fixation_y[start_idx:end_idx + 1] = fix_y
                event_duration[start_idx:end_idx + 1] = window_duration_seconds * 1000  # Convert back to ms
                fixation_ids[start_idx:end_idx + 1] = fixation_id
                fixation_id += 1

            start_idx = end_idx + 1 if current_idx > start_idx else start_idx + 1

        # Assign computed values to the DataFrame
        result_data['event_type'] = event_type
        result_data['fixation_x'] = fixation_x
        result_data['fixation_y'] = fixation_y
        result_data['event_duration'] = event_duration
        result_data['fixation_id'] = fixation_ids

        return result_data

    def detect_event(self, plot=False, min_fixation_duration=50.0, aois=None, algorithm='idt',
                    dispersion_threshold=150.0, optimize_threshold=False, adapt_velocity=False):
        """Detects events in the loaded gaze data using the specified algorithm.

        Args:
            plot (bool): Whether to plot the event segmentation.
            min_fixation_duration (float): Minimum duration for a fixation (in milliseconds).
            aois (pd.DataFrame): Areas of interest for AOI classification.
            algorithm (str): Event detection algorithm ('velocity' or 'idt').
            dispersion_threshold (float): Dispersion threshold (for 'idt' algorithm, in pixels).
            adapt_velocity:
            optimize_threshold:

        Returns:
            pd.DataFrame: Processed gaze data with event classifications.

        """
        if not self.is_valid_data:
            return None

        try:
            # if algorithm == 'idt':
            #     data = self.classify_idt(self.gaze_data, dispersion_threshold=dispersion_threshold,
            #                            min_fixation_duration=min_fixation_duration)
            # else:
            #     raise ValueError(f"Unsupported algorithm: {algorithm}")
            # First detect events using the original algorithm
            if algorithm == 'idt':
                if optimize_threshold:
                    best_thresh = self.optimize_threshold(self.gaze_data, adapt_velocity=adapt_velocity)
                    print(f"Best Threshold:{best_thresh}")
                    data = self.classify_idt(self.gaze_data, dispersion_threshold=best_thresh,
                                             min_fixation_duration=min_fixation_duration, adapt_velocity=adapt_velocity)
                else:
                    data = self.classify_idt(self.gaze_data, dispersion_threshold=dispersion_threshold,
                                             min_fixation_duration=min_fixation_duration, adapt_velocity=adapt_velocity)
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")

        except Exception as e:
            logging.error(f"Error detecting events: {e}")
            return None

        if aois is not None:
            data = self.classify_aoi(data, aois)

        return data

    def classify_aoi(self, gaze_data, aois, algorithm='weighted_bbox_attach'):
        """
        Checks if the fixation points are in the AOI or not.

        Args:
            gaze_data (pd.DataFrame): DataFrame containing gaze data with fixation coordinates.
            aois (pd.DataFrame): DataFrame containing AOI definitions.
            algorithm (str): Algorithm to use for AOI classification ('standard' or 'attach').
            - 'standard': Checks if fixation is within AOI boundaries.
            - 'attach': Attaches each fixation to its closest AOI.

        Returns:
            pd.DataFrame: Gaze data with added AOI classification.
        """
        if not self.is_valid_data:
            return None, None

        # first get all unique fixation points
        fixations = gaze_data[['fixation_id', 'fixation_x', 'fixation_y']].drop_duplicates()

        # remove NaN values
        fixations = fixations[fixations['fixation_x'].notna() & fixations['fixation_y'].notna()]

        fixations['aoi_type'] = np.nan
        fixations['aoi'] = np.nan

        # reset index
        fixations.reset_index(drop=True, inplace=True)

        if algorithm == 'standard':
            # Standard algorithm: check if fixation is within AOI boundaries
            for index, row in fixations.iterrows():
                for aoi_index, aoi in aois.iterrows():
                    # aois are in format ['aoi_type', 'aoi', 'pos_x', 'pos_y', 'width', 'height'']
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

                    # Calculate weighted distance
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

    def merge_close_fixations(self, gaze_data, distance_threshold=100.0):
        """
        Merges consecutive fixations that are close to each other.

        Args:
            gaze_data (pd.DataFrame): DataFrame containing gaze data with event classifications.
            distance_threshold (float): Maximum distance (in pixels) between consecutive fixations to be merged.

        Returns:
            pd.DataFrame: Gaze data with merged fixations.
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
            # Find the corresponding merged fixation
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

    def detect_event_with_merge(self, plot=False, min_fixation_duration=50.0, aois=None,
                               algorithm='idt', dispersion_threshold=150.0, merge_distance=100.0, adapt_velocity=False,
                               optimize_threshold=False, return_threshold=False):
        """
        Detects events in the loaded gaze data using the specified algorithm and merges close fixations.

        Args:
            plot (bool): Whether to plot the event segmentation.
            min_fixation_duration (float): Minimum duration for a fixation (in milliseconds).
            aois (pd.DataFrame): Areas of interest for AOI classification.
            algorithm (str): Event detection algorithm ('velocity' or 'idt').
            dispersion_threshold (float): Dispersion threshold (for 'idt' algorithm, in pixels).
            merge_distance (float): Maximum distance for merging consecutive fixations.
            adapt_velocity (bool): Whether to adapt the dispersion threshold based on velocity.
            optimize_threshold (bool): Whether to optimize the dispersion threshold.
            return_threshold (bool): Whether to return the optimized threshold.

        Returns:
            pd.DataFrame: Processed gaze data with event classifications and merged fixations.
        """
        if not self.is_valid_data:
            return None

        try:
            # First detect events using the original algorithm
            if algorithm == 'idt':
                if optimize_threshold:
                    best_thresh = self.optimize_threshold(self.gaze_data, adapt_velocity=adapt_velocity)
                    print(f"Best Threshold:{best_thresh}")
                    data = self.classify_idt(self.gaze_data, dispersion_threshold=best_thresh,
                                           min_fixation_duration=min_fixation_duration, adapt_velocity=adapt_velocity)
                else:
                    data = self.classify_idt(self.gaze_data, dispersion_threshold=dispersion_threshold,
                                           min_fixation_duration=min_fixation_duration, adapt_velocity=adapt_velocity)
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")

            # Merge close fixations
            data = self.merge_close_fixations(data, distance_threshold=merge_distance)

            if aois is not None:
                data = self.classify_aoi(data, aois)

        except Exception as e:
            logging.error(f"Error detecting events: {e}")
            return None

        if return_threshold:
            return data, best_thresh

        return data

    def process_event_with_merge(self, output_dir, plot=True,
                                min_fixation_duration=50.0, aoi_file_path=None, algorithm='idt',
                                merge_distance:float = None, dispersion_threshold=150.0, adapt_velocity=False,
                                optimize_threshold=False, duration_cutoff: float = None):
        """
        Processes event detection for a single user and image with fixation merging and stores the output.

        Args:
            output_dir (str): Directory to store the output CSV files.
            plot (bool): Whether to plot the event segmentation.
            min_fixation_duration (float): Minimum duration for fixation events.
            aoi_file_path (str): Whether to check if the fixation points are in the AOI or not
            algorithm: Event detection algorithm ('ivt' or 'idt').
            merge_distance (float): Maximum distance for merging consecutive fixations.
            dispersion_threshold (float): Dispersion threshold for the I-DT algorithm.
            adapt_velocity (bool): Whether to adapt the dispersion threshold based on velocity.
            optimize_threshold (bool): Whether to optimize the dispersion threshold.
            duration_cutoff (float): Optional duration cutoff.

        Returns:
            bool: True if processing was successful, False otherwise.
        """
        aois = None
        if aoi_file_path is not None:
            aois = pd.read_csv(aoi_file_path)

        # If a duration cutoff is specified, filter the gaze data
        if duration_cutoff is not None:
            total_duration = self.gaze_data["timestamp"].iloc[-1]
            if total_duration > duration_cutoff:
                start_threshold = total_duration - duration_cutoff
                self.gaze_data = self.gaze_data.loc[self.gaze_data["timestamp"] >= start_threshold].reset_index(
                    drop=True)

                # rebase timestamps so t(0) == 0
                t0 = self.gaze_data["timestamp"].iloc[0]
                self.gaze_data["timestamp"] = self.gaze_data["timestamp"] - t0

        # Use the new method with merging
        if merge_distance:
            event_gaze = self.detect_event_with_merge(
                plot=plot,
                min_fixation_duration=min_fixation_duration,
                aois=aois,
                algorithm=algorithm,
                merge_distance=merge_distance,
                dispersion_threshold=dispersion_threshold,
                adapt_velocity=adapt_velocity,
                optimize_threshold=optimize_threshold
            )
        else:
            event_gaze = self.detect_event(
                plot=plot,
                min_fixation_duration=min_fixation_duration,
                aois=aois,
                algorithm=algorithm,
                dispersion_threshold=dispersion_threshold,
            )

        if event_gaze is not None:
            event_gaze = self.clean_fixations(event_gaze)
            event_gaze.to_csv(output_dir, index=False, sep=';')
            logging.info(f"Processed and saved merged event data in {output_dir}")
            return True
        else:
            logging.error(f"Failed to process merged event data")
            return False

    def clean_fixations(self, events_df):
        # events_df["fixation_x"] /= 640
        # events_df["fixation_y"] /= 480
        events_df.rename(columns={"fixation_x": "norm_pos_x", "fixation_y": "norm_pos_y", "event_duration": "duration",
                                "fixation_id": "id"}, inplace=True)
        events_df.drop(columns=["Unnamed: 4", "merged", "x", "y"], inplace=True, errors='ignore')
        events_df['video_frame_count'] = events_df['video_frame_count'].astype('Int64')  # nullable integer if there are nulls

        # --- (B) Compute start/end frame for each fixation_id (exclude null fixation_id) ---
        fix_bounds = (
            events_df.dropna(subset=['id'])  # ignore rows without a fixation_id
            .groupby('id', dropna=False)['video_frame_count']
            .agg(start_frame_index='min', end_frame_index='max')
            .reset_index()
        )

        # --- (C) Merge the start/end back to the original dataframe ---
        events_df = events_df.merge(fix_bounds, on='id', how='left')

        # Now every row that has a fixation_id will have start_frame_index and end_frame_index
        # Rows with NaN fixation_id will have NaN in those two new columns.

        # --- (D) Drop duplicate fixation_id rows, keep the first occurrence ---
        # If you want the first occurrence in temporal order, ensure dataframe is sorted by timestamp/frame first:
        events_df = events_df.sort_values(['video_frame_count', 'timestamp']).reset_index(drop=True)
        events_df = events_df[events_df["event_type"] == "Fixation"].reset_index(drop=True)

        # Drop duplicates (keeps the first row for each fixation_id)
        events_df = events_df.drop_duplicates(subset=['id'], keep='first').reset_index(drop=True)

        return events_df

    def optimize_threshold(self, gaze_data, min_fixation_duration=50, adapt_velocity=False, candidate_thresholds=None):
        """
        Optimize dispersion threshold using cluster validity metrics.

        Args:
            gaze_data: DataFrame with ['x', 'y', 'timestamp']
            candidate_thresholds: List of dispersion thresholds to test
            min_fixation_duration: Minimum fixation duration in ms
            adapt_velocity: Whether to adapt the dispersion threshold based on velocity

        Returns:
            Best threshold according to Calinski-Harabasz score
        """
        if candidate_thresholds is None:
            candidate_thresholds = [125, 150, 175, 200, 225, 250, 275, 300]

        best_score = -np.inf
        best_threshold = candidate_thresholds[0]

        for threshold in candidate_thresholds:
            try:
                # Run I-DT classification
                classified_df = self.classify_idt(
                    gaze_data.copy(),
                    dispersion_threshold=threshold,
                    min_fixation_duration=min_fixation_duration,
                    adapt_velocity=adapt_velocity
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

if __name__ == '__main__':
    folders = ["7"] # [f"{i}" for i in range(1, 7)]
    file_name = "gaze.csv"

    for folder in folders:
        print(f"Processing folder: {folder}")

        if folder == "3":
            # Skip folder 3 as it has no data
            continue

        gaze_file = pd.read_csv(f"Dataset/{folder}/{file_name}", delimiter=";")
        event_detection = EventDetection(gaze_file)

        # Process with IDT algorithm
        event_detection.process_event_with_merge(output_dir=f"Dataset/{folder}/fixations.csv",
                                                plot=False,
                                                min_fixation_duration=50.0,
                                                merge_distance=None,
                                                dispersion_threshold=25.0,
                                                adapt_velocity=False,
                                                optimize_threshold=False,
                                                algorithm="idt")

    print("Processing complete!")
