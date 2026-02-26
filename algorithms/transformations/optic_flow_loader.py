"""Provides a transformation for loading pre-computed optical flow data."""

from dataclasses import astuple

import dask.array as da
import numpy as np
from gaze_dataset import GazeDataset
from tqdm.auto import tqdm

from .transformation import Transformation


class OpticFlowLoader(Transformation):
    """Loads pre-computed optical flow data and adds it to the dataset.

    This transformation loads a potentially large optical flow .npy file,
    calculates the mean flow vector for each frame, and adds this mean flow
    (x and y components) to the corresponding rows in the dataset's DataFrames.

    The calculated mean flow is cached to a separate file to speed up
    subsequent loads.
    """

    def apply(self, data: GazeDataset):
        """Apply the optical flow loading to all splits in the dataset.

        Args:
            data: The `GazeDataset` to be transformed.

        """
        for params, split in tqdm(data.get_splits(), desc="Loading Optic Flow"):
            flow_path = data.get_flow_path(params)
            mean_path = flow_path.parent / f"mean_flow_{'_'.join(map(str, astuple(params)))}.npy"
            if mean_path.exists():
                with mean_path.open("rb") as f:
                    mean_flow = np.load(f)
                    assert mean_flow.shape[1] == 2
                    split[["flow_x", "flow_y"]] = mean_flow
                    continue

            flow = da.from_array(np.load(flow_path, mmap_mode="r"))
            average_flow = da.nanmean(flow, axis=(1, 2), dtype=np.float64).compute()

            frames = split["frame"].astype(int).to_numpy()
            frames = np.clip(frames, 0, average_flow.shape[0] - 1)
            split[["flow_x", "flow_y"]] = average_flow[frames]
            with mean_path.open("wb") as f:
                np.save(f, split[["flow_x", "flow_y"]].to_numpy())
