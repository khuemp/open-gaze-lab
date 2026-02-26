"""Provides a transformation for loading pre-computed patch similarity data."""

import numpy as np
import pandas as pd
from gaze_dataset import GazeDataset

from .transformation import Transformation


class PatchSimilarityLoaderDrews(Transformation):
    """Loads pre-computed patch similarity data for the Drews dataset.

    This transformation reads visual similarity scores and their corresponding
    timestamps from .npy files and merges them into the main data split
    DataFrame. The merge is performed based on the closest timestamp, using
    `pd.merge_asof`.
    """

    def apply(self, data: GazeDataset):
        """Apply the patch similarity loading to all splits in the dataset.

        Args:
            data: The `GazeDataset` to be transformed.

        """
        for param, split in data.get_splits():
            sim_times = np.load(
                data.get_similarity_path(param).parent / "time_visual_similarity.npy",
            )
            sim = np.load(data.get_similarity_path(param).parent / "visual_similarity.npy")
            sim_df = pd.DataFrame({"sim_time": sim_times, "patch_dist": sim})

            merged = pd.merge_asof(
                split,
                sim_df,
                left_on="timestamp",
                right_on="sim_time",
                direction="forward",
            )
            merged.fillna(0, inplace=True)
            split["patch_dist"] = merged["patch_dist"]
