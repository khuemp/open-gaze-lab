
from .pipeline import *
from .detection_algorithms import *
from .utils import *
from .visualization import plot_gaze_points_and_fixations, plot_gaze_with_time_scrolling

class EventDetection:
    """
    Class for detecting events in gaze data.

    Args:
        dataset_loader (DatasetLoaderV2): DatasetLoaderV2 instance to load gaze data.

    Methods:
        load_gaze_data(user, image): Loads gaze data for the specified user and image.
        detect_event(detect_plot=False): Detects events in the loaded gaze data and returns a tuple of discrete time intervals and class labels.
    """

    def __init__(self, loaded_gaze_df, resolution=(2560, 1440), column_mapping=None, is_normalized=None):
        """
        Initializes the EventDetection class.
        
        Args:
            loaded_gaze_df: DataFrame containing gaze data with timestamp column
            resolution: Tuple of (width, height) for screen resolution, defaults to (2560, 1440)
            column_mapping: Dict mapping standard names ('x', 'y', 'timestamp') to actual column names.
                           If None, assumes standard column names ('x', 'y', 'timestamp').
            is_normalized: Whether coordinates are normalized (0-1). If None, auto-detects.
        """
        self.gaze_data = loaded_gaze_df.copy()
        self.is_valid_data = True
        
        # Default column mapping
        if column_mapping is None:
            column_mapping = {'x': 'x', 'y': 'y', 'timestamp': 'timestamp'}
        
        # Rename columns to standard names if needed
        rename_map = {}
        if column_mapping.get('x') != 'x' and column_mapping.get('x') in self.gaze_data.columns:
            rename_map[column_mapping['x']] = 'x'
        if column_mapping.get('y') != 'y' and column_mapping.get('y') in self.gaze_data.columns:
            rename_map[column_mapping['y']] = 'y'
        if column_mapping.get('timestamp') != 'timestamp' and column_mapping.get('timestamp') in self.gaze_data.columns:
            rename_map[column_mapping['timestamp']] = 'timestamp'
        
        if rename_map:
            self.gaze_data = self.gaze_data.rename(columns=rename_map)
        
        # Auto-detect normalization if not provided
        if is_normalized is None:
            x_values = self.gaze_data['x'].dropna()
            y_values = self.gaze_data['y'].dropna()
            if len(x_values) > 0 and len(y_values) > 0:
                x_max = x_values.max()
                y_max = y_values.max()
                # If max values > 2, assume pixel coordinates
                is_normalized = not (x_max > 2 or y_max > 2)
            else:
                is_normalized = True  # Default to normalized if no data
        
        # Scale normalized coordinates to screen resolution (only if normalized)
        if is_normalized:
            self.gaze_data['x'] *= resolution[0]
            self.gaze_data['y'] *= resolution[1]

        # Convert timestamps to milliseconds if needed
        first_timestamp = self.gaze_data['timestamp'].iloc[0]
        if first_timestamp > 1000:  # If timestamps are in seconds (either epoch or regular)
            self.gaze_data['timestamp'] = (self.gaze_data['timestamp'] - first_timestamp) * 1000
        elif self.gaze_data['timestamp'].iloc[-1] < 100:  # Small values suggest seconds from start
            # Convert from seconds to milliseconds
            self.gaze_data['timestamp'] = self.gaze_data['timestamp'] * 1000
        
        # Initialize logging with timestamp and level information
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
        )

        # Bind imported functions as instance methods for object-oriented interface
        self.detect_event = detect_event.__get__(self)
        self.classify_aoi = classify_aoi.__get__(self)
        self.process_event = process_event.__get__(self)


class EyeTrackingVisualizer:
    """
    Class for visualizing eye-tracking data.

    Methods:
        plot_gaze_points_and_fixations(gaze_data, bg_image_path=None, aois=None, show_attach=True, attach_type='bbox'):
            Visualizes gaze points and fixations from eye-tracking data using Plotly.
        plot_gaze_with_time_scrolling(output_dir, bg_image_path=None, aois=None, time_window_ms=5000, step_ms=100):
            Creates an interactive time-scrollable visualization of gaze data and fixations.
    """

    def __init__(self, loaded_event_df):
        """Initializes the EyeTrackingVisualizer class."""
        # Set up logging system with detailed format
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
        )
        # Create independent copy of event data to prevent modifications
        self.event_data_df = loaded_event_df.copy()

        # Attach visualization functions as instance methods
        self.plot_gaze_points_and_fixations = plot_gaze_points_and_fixations.__get__(self)
        self.plot_gaze_with_time_scrolling = plot_gaze_with_time_scrolling.__get__(self)

__all__ = ["EventDetection", "EyeTrackingVisualizer"]
