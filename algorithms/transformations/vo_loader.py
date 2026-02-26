"""Provides a transformation for loading pre-computed visual odometry data."""

import numpy as np
from gaze_dataset import GazeDataset
from tqdm import tqdm

from constants import PROJECT_ROOT
from datasets import GIWParams

from .transformation import Transformation


class VOLoader(Transformation):
    """Loads pre-computed visual odometry (VO) data and adds it to the dataset.

    This transformation reads VO data, which includes head rotation and body
    motion, from text files. It calculates the speed of head rotation and body
    motion and adds these as new 'head_speed' and 'body_speed' columns to the
    corresponding data splits.
    """

    def apply(self, data: GazeDataset):
        """Apply the VO loading and processing to all splits in the dataset.

        Args:
            data: The `GazeDataset` to be transformed.

        """
        for params, split in tqdm(data.get_splits(), desc="Loading visual odometry"):
            params: GIWParams = params
            base_path = (
                PROJECT_ROOT
                / f"dependencies/ACE-DNV/GiW/{params.participant_id}/{params.trial_id}"
            )

            visodom = np.loadtxt(base_path / "visOdom.txt", delimiter=" ")
            head_rotation = visodom[:, 5]
            body_motion = visodom[:, 1:3]

            start_frame, end_frame = np.loadtxt(base_path / "range.txt")
            split.drop(
                split[
                    ~split["frame"].between(
                        start_frame,
                        start_frame + len(head_rotation) + 1,
                        inclusive="neither",
                    )
                ].index,
                inplace=True,
            )
            split.reset_index(inplace=True, drop=True)

            frames, indices = np.unique(split["frame"], return_index=True)

            head_rotation = head_rotation[frames[:-1] - int(start_frame)]

            diffs = head_rotation[:-1] - head_rotation[1:]

            amplitudes = np.sqrt(diffs**2)
            speeds = amplitudes / data.sample_duration_ms * 1000
            angles = np.arctan(diffs)
            split["head_speed"] = 0.0
            split["head_angle"] = 0.0
            split.loc[indices[1:-1], "head_speed"] = speeds
            split.loc[indices[1:-1], "head_angle"] = angles

            body_motion = body_motion[frames[:-1] - int(start_frame)]
            dists = np.linalg.norm(body_motion[:-1] - body_motion[1:], axis=1)
            speeds = dists / data.sample_duration_ms * 1000
            speeds = np.concatenate([speeds, [speeds[-1]]])

            speeds = np.clip(speeds, a_min=None, a_max=np.quantile(speeds, 0.75))
            speeds = (speeds - np.min(speeds)) / (np.max(speeds) - np.min(speeds))
            split["body_speed"] = 0.0
            split.loc[indices[1:], "body_speed"] = speeds
