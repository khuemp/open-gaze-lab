
from .pipeline import *
from .detection_algorithms import *
from .utils import *

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


        # Bind external functions as methods
        self.detect_event = detect_event.__get__(self)
        self.detect_event_with_merge = detect_event_with_merge.__get__(self)
        self.classify_aoi = classify_aoi.__get__(self)
        self.process_event = process_event.__get__(self)

__all__ = ["EventDetection"]
