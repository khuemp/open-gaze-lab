
from .pipeline import *
from .detection_algorithms import *
from .utils import *
from .visualization import plot_gaze_points_and_fixations

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

        # Configure logger
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
        )


        # Bind external functions as methods
        self.detect_event = detect_event.__get__(self)
        self.classify_aoi = classify_aoi.__get__(self)
        self.process_event = process_event.__get__(self)


class EyeTrackingVisualizer:
    """
    Class for visualizing eye-tracking data.

    Methods:
        plot_gaze_points_and_fixations(gaze_data, bg_image_path=None, aois=None, show_attach=True, attach_type='bbox'):
            Visualizes gaze points and fixations from eye-tracking data using Plotly.
    """

    def __init__(self, loaded_event_df):
        """Initializes the EyeTrackingVisualizer class."""
        # Configure logger
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
        )
        self.event_data_df = loaded_event_df.copy()

        # Bind external functions as methods
        self.plot_gaze_points_and_fixations = plot_gaze_points_and_fixations.__get__(self)

__all__ = ["EventDetection", "EyeTrackingVisualizer"]
