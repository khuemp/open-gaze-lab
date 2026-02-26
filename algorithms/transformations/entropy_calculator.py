"""Provides a transformation for calculating entropy-based features from optical flow."""

from dataclasses import astuple

import numpy as np
from gaze_dataset import GazeDataset
from scipy.stats import entropy
from tqdm.auto import tqdm

from .transformation import Transformation


class EntropyCalculator(Transformation):
    """Calculates entropy features from optical flow patches around gaze points.

    For each gaze point, this transformation extracts a patch of the
    corresponding optical flow field. It then computes the entropy of the
    flow magnitudes, the entropy of the flow angles, and the joint entropy
    of magnitudes and angles within that patch.

    The calculated features are cached to a .npy file to speed up subsequent
    runs.

    Attributes:
        EPSILON: A small value to prevent division by zero and log(0).
        radius: The radius of the square patch to extract around the gaze point.

    """

    EPSILON = 1e-5

    def __init__(self, radius: int = 100):
        """Initialise the EntropyCalculator.

        Args:
            radius: The radius of the square patch (side length = 2 * radius)
                    to extract around each gaze point.

        """
        self.radius = radius
        super().__init__()

    def apply(self, data: GazeDataset):
        """Apply the entropy calculations to all splits in the dataset.

        Args:
            data: The `GazeDataset` to be transformed.

        """
        for params, split in tqdm(data.get_splits(), desc="Calculating entropies"):
            flow_path = data.get_flow_path(params)
            entropy_path = (
                flow_path.parent / f"entropies_{'_'.join(map(str, astuple(params)))}.npy"
            )
            if entropy_path.exists():
                with entropy_path.open("rb") as file:
                    entropies = np.load(file)
                    assert entropies.shape[1] == 3  # noqa: PLR2004
                    split[["mag_entropy", "angle_entropy", "joint_entropy"]] = entropies
                    continue

            flow = np.load(flow_path, mmap_mode="r")

            # split["frame"].dropna(inplace=True)
            frames = split["frame"].astype(int).to_numpy()

            unique_frames, unique_indices = np.unique(frames, return_index=True)
            centers_x = np.clip(split.loc[unique_indices, "filter_x"], 0, data.width).to_numpy()
            centers_y = np.clip(split.loc[unique_indices, "filter_y"], 0, data.height).to_numpy()

            mag_entropies = np.zeros(len(unique_indices))
            angle_entropies = np.zeros(len(unique_indices))
            joint_entropies = np.zeros(len(unique_indices))
            mag_bins = np.linspace(0, 100, 21)
            angle_bins = np.linspace(-np.pi, np.pi, 37)

            min_available_values = min(centers_x.shape[0], flow.shape[0])
            if min_available_values + unique_frames[0] >= flow.shape[0]:
                min_available_values -= min_available_values + unique_frames[0] - flow.shape[0]

            for i in tqdm(
                range(min_available_values),
                desc="Calculating entropies",
                leave=True,
                position=0,
            ):
                cx = centers_x[i]
                cy = centers_y[i]
                # Calculate boundaries
                y_start, y_end = cy - self.radius, cy + self.radius
                x_start, x_end = cx - self.radius, cx + self.radius

                # Clip coordinates to be valid image indices
                y1, y2 = max(0, y_start), min(data.height, y_end)
                x1, x2 = max(0, x_start), min(data.width, x_end)

                flow_width = flow.shape[2]
                flow_height = flow.shape[1]

                # Scale points to the saved optical flow resolution
                x1 = int(x1 * flow_width / data.width)
                x2 = int(x2 * flow_width / data.width)
                y1 = int(y1 * flow_height / data.height)
                y2 = int(y2 * flow_height / data.height)

                # Extract the valid part from the image
                # Flow has shape [n_frames, HEIGHT, WIDTH, 2]
                patch = flow[i + unique_frames[0], y1:y2, x1:x2, :]

                if patch.size == 0 or np.ptp(patch) == 0:
                    mag_entropies[i] = 0.0
                    angle_entropies[i] = 0.0
                    joint_entropies[i] = 0.0
                    continue

                u = patch[..., 0].astype(np.float64)
                v = patch[..., 1].astype(np.float64)

                # patch has shape [HEIGHT, WIDTH, 2]
                magnitudes = np.hypot(u, v).flatten()
                angles = np.arctan2(v, u).flatten()

                mask = np.isfinite(magnitudes) & (magnitudes > self.EPSILON)
                valid_angles = angles[mask]
                valid_magnitudes = magnitudes[mask]

                if valid_magnitudes.size == 0 or np.ptp(valid_magnitudes) == 0:
                    mag_entropies[i] = 0.0
                else:
                    counts, _ = np.histogram(
                        valid_magnitudes.flatten(),
                        bins=mag_bins,
                        density=True,
                    )
                    mag_entropies[i] = entropy(counts)

                if valid_angles.size == 0 or np.ptp(valid_angles) == 0:
                    angle_entropies[i] = 0.0
                else:
                    counts, _ = np.histogram(valid_angles.flatten(), bins=angle_bins, density=True)
                    angle_entropies[i] = entropy(counts)

                if valid_angles.size == 0 or valid_magnitudes.size == 0:
                    joint_entropies[i] = 0.0
                else:
                    counts, _, _ = np.histogram2d(
                        valid_magnitudes,
                        valid_angles,
                        bins=[mag_bins, angle_bins],
                        density=True,
                    )
                    joint_entropies[i] = entropy(counts.flatten())

            split["mag_entropy"] = np.nan
            split["angle_entropy"] = np.nan
            split["joint_entropy"] = np.nan
            split.loc[unique_indices, "mag_entropy"] = mag_entropies
            split.loc[unique_indices, "angle_entropy"] = angle_entropies
            split.loc[unique_indices, "joint_entropy"] = joint_entropies
            split.interpolate(method="cubic", inplace=True)
            full_entropies = split[["mag_entropy", "angle_entropy", "joint_entropy"]].to_numpy()
            with entropy_path.open("wb") as file:
                np.save(file, full_entropies)
