"""Provides a transformation for applying a Savitzky-Golay filter to gaze data."""

from scipy.signal import savgol_filter

from datasets import GazeDataset

from .transformation import Transformation


class SavgolFilter(Transformation):
    """Applies a Savitzky-Golay filter to smooth gaze coordinate data.

    This transformation adds 'filter_x' and 'filter_y' columns to the dataset's
    DataFrames, containing the smoothed gaze coordinates.

    Attributes:
        window_size_ms: The desired window size for the filter in milliseconds.
        sample_rate_hz: The sample rate of the data, used to convert the
                        window size from ms to number of samples.

    """

    def __init__(self, window_size_ms: int, sample_rate_hz: int):
        self.window_size_ms = window_size_ms
        self.sample_rate_hz = sample_rate_hz

    @property
    def window_size(self) -> int:
        """Calculate the window size in samples from the duration in ms."""
        frame_duration_ms = 1 / self.sample_rate_hz * 1000
        return int(self.window_size_ms // frame_duration_ms)

    def apply(self, data: GazeDataset):
        """Apply the Savitzky-Golay filter to all splits in the dataset.

        Args:
            data: The `GazeDataset` to be transformed.

        """
        for _key, split in data.get_splits():
            split.fillna(0, inplace=True)
            split["filter_x"] = savgol_filter(
                split["x"],
                window_length=self.window_size,
                polyorder=3,
            )
            split["filter_y"] = savgol_filter(
                split["y"],
                window_length=self.window_size,
                polyorder=3,
            )
