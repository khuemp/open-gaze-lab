"""OpenGazeLab event-detection backend package.

Public surface (what ``main.py`` imports):

* :class:`EventDetection` — orchestrates fixation/saccade detection.
* :class:`EyeTrackingVisualizer` — wraps the visualization plotters.
* CSV helpers — :func:`load_csv_gaze_data`, :func:`detect_column_mapping`, etc.
* Head-mounted helpers — :func:`load_npy_dataset`, :func:`extract_video_metadata`.
* :func:`generate_video_gaze_visualization` — head-mounted overlay video.
"""

import logging

from .pipeline import EventDetection
from .preprocess_csv import (
    detect_column_mapping,
    detect_delimiter,
    detect_if_normalized,
    load_csv_gaze_data,
)
from .preprocess_headmounted import (
    extract_video_metadata,
    load_giw_dataset,
    load_head_mounted_dataset,
    load_npy_dataset,
)
from .visualization import (
    generate_video_gaze_visualization,
    plot_gaze_points_and_fixations,
    plot_gaze_with_time_scrolling,
)


class EyeTrackingVisualizer:
    """Thin wrapper that binds the visualization plotters to a result DataFrame.

    Args:
        loaded_event_df: Detection-output DataFrame (from
            :meth:`EventDetection.process_event`).
        resolution: ``(width, height)`` of the recording / screen in pixels.
    """

    def __init__(self, loaded_event_df, resolution):
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            level=logging.INFO,
        )
        self.event_data_df = loaded_event_df.copy()
        self.resolution = resolution

        # Bind visualization functions as instance methods.
        self.plot_gaze_points_and_fixations = plot_gaze_points_and_fixations.__get__(self)
        self.plot_gaze_with_time_scrolling = plot_gaze_with_time_scrolling.__get__(self)


__all__ = [
    "EventDetection",
    "EyeTrackingVisualizer",
    # CSV preprocessing
    "load_csv_gaze_data",
    "detect_delimiter",
    "detect_column_mapping",
    "detect_if_normalized",
    # Head-mounted preprocessing
    "load_npy_dataset",
    "load_giw_dataset",
    "load_head_mounted_dataset",
    "extract_video_metadata",
    # Visualization
    "generate_video_gaze_visualization",
]
