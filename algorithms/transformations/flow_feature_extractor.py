"""Provides a transformation for extracting experimental optical flow features.

This module defines the `FlowFeatureExtractor`, which calculates several novel
features based on the characteristics of optical flow in patches around the
gaze point.
"""

from dataclasses import astuple

import numpy as np
from gaze_dataset import GazeDataset
from skimage.segmentation import flood
from sklearn.cluster import KMeans
from tqdm.auto import tqdm

from .transformation import Transformation


class FlowFeatureExtractor(Transformation):
    """Calculates experimental features from optical flow patches.

    This transformation extracts a patch of optical flow around each gaze point
    and computes three experimental features designed to capture the uniformity
    of flow within that patch:
    1.  `f_distance`: The absolute difference between the mean magnitude of an
        inner patch and the mean magnitude of the whole patch.
    2.  `f_cluster`: The absolute difference in mean magnitudes between two
        clusters of flow vectors found via K-Means clustering.
    3.  `f_fill`: The absolute difference in mean magnitudes between a gaze-central,
        flood-filled region and the surrounding region.

    The calculated features are cached to a file to accelerate subsequent runs.
    """

    EPSILON = 1e-5
    TOLERANCE = 0.01

    def __init__(self, radius: int = 200):
        """Initialise the FlowFeatureExtractor.

        Args:
            radius: The radius of the square patch (side length = 2 * radius)
                    to extract around each gaze point.

        """
        self.radius = radius
        super().__init__()

    def extract_patch(
        self,
        flow: np.ndarray,
        radius: int,
        x: int,
        y: int,
        data: GazeDataset,
        frame: int,
    ) -> np.ndarray:
        """Extract a square patch from the optical flow field.

        Args:
            flow: The full optical flow tensor for the dataset.
            radius: The radius of the patch to extract.
            x: The center x-coordinate of the patch (in original video resolution).
            y: The center y-coordinate of the patch (in original video resolution).
            data: The GazeDataset object, used for resolution metadata.
            frame: The frame index from which to extract the patch.

        Returns:
            The extracted patch as a NumPy array.

        """
        y_start, y_end = y - radius, y + radius
        x_start, x_end = x - radius, x + radius

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
        return flow[frame, y1:y2, x1:x2, :]

    def apply(self, data: GazeDataset):
        """Apply the feature extraction to all splits in the dataset.

        Args:
            data: The `GazeDataset` to be transformed.

        """
        for params, split in tqdm(data.get_splits(), desc="Calculating flow features"):
            flow_path = data.get_flow_path(params)
            feature_path = flow_path.parent / f"features_{'_'.join(map(str, astuple(params)))}.npy"
            if feature_path.exists():
                with feature_path.open("rb") as file:
                    features = np.load(file)
                    assert features.shape[1] == 3  # noqa: PLR2004
                    split[["f_distance", "f_cluster", "f_fill"]] = features
                    continue

            flow = np.load(flow_path, mmap_mode="r")

            frames = split["frame"].astype(int).to_numpy()

            unique_frames, unique_indices = np.unique(frames, return_index=True)
            centers_x = np.clip(split.loc[unique_indices, "filter_x"], 0, data.width).to_numpy()
            centers_y = np.clip(split.loc[unique_indices, "filter_y"], 0, data.height).to_numpy()

            dists = np.zeros(len(unique_indices))
            clusters = np.zeros(len(unique_indices))
            fills = np.zeros(len(unique_indices))

            min_available_values = min(centers_x.shape[0], flow.shape[0])
            if min_available_values + unique_frames[0] >= flow.shape[0]:
                min_available_values -= min_available_values + unique_frames[0] - flow.shape[0]

            for i in tqdm(
                range(min_available_values),
                desc="Going through frames",
                leave=True,
                position=0,
            ):
                cx = centers_x[i]
                cy = centers_y[i]
                frame = i + unique_frames[0]
                patch = self.extract_patch(
                    flow,
                    radius=self.radius,
                    x=cx,
                    y=cy,
                    data=data,
                    frame=frame,
                )
                inner_patch = self.extract_patch(
                    flow,
                    radius=int(self.radius / 2),
                    x=cx,
                    y=cy,
                    data=data,
                    frame=frame,
                )

                if patch.size == 0 or np.ptp(patch) == 0:
                    dists[i] = 0.0
                    clusters[i] = 0.0
                    fills[i] = 0.0
                    continue

                u = patch[..., 0].astype(np.float64)
                v = patch[..., 1].astype(np.float64)

                # patch has shape [HEIGHT, WIDTH, 2]
                magnitudes = np.hypot(u, v)
                magnitudes = np.nan_to_num(magnitudes)

                if magnitudes.size == 0 or np.ptp(magnitudes) == 0:
                    dists[i] = 0.0
                    clusters[i] = 0.0
                    fills[i] = 0.0
                    continue

                ### Different features calculations
                # 1. Distance based difference
                u_inner = inner_patch[..., 0].astype(np.float64)
                v_inner = inner_patch[..., 1].astype(np.float64)

                magnitudes_inner = np.hypot(u_inner, v_inner).flatten()
                magnitudes_inner = np.nan_to_num(magnitudes_inner)

                # Should we use all magnitues or filter here again with
                # outer_magnitues = magnitues without the inner ones
                dists[i] = abs(magnitudes_inner.mean() - magnitudes.mean())

                # 2. Clustering based difference
                m_mag = np.array(
                    [
                        [magnitudes[x, y], x, y]
                        for x in range(magnitudes.shape[0])
                        for y in range(magnitudes.shape[1])
                    ],
                )
                m_mag_normed = m_mag / m_mag.max(axis=0)
                cluster_indices = KMeans(n_clusters=2).fit_predict(m_mag_normed)

                indices0 = np.where(cluster_indices == 0)[0]
                indices1 = np.where(cluster_indices == 1)[0]

                mag0 = m_mag[indices0][:, 0]
                mag1 = m_mag[indices1][:, 0]

                clusters[i] = abs(mag0.mean() - mag1.mean())

                # 3. Flood Fill based difference
                cx = int(patch.shape[0] // 2)
                cy = int(patch.shape[1] // 2)
                mag_normed = magnitudes / np.mean(magnitudes)
                mask = flood(mag_normed, (cx, cy), tolerance=self.TOLERANCE)

                mag0 = magnitudes[mask]
                mag1 = magnitudes[~mask]

                fills[i] = abs(mag0.mean() - mag1.mean())

            split["f_dist"] = np.nan
            split["f_cluster"] = np.nan
            split["f_fill"] = np.nan
            split.loc[unique_indices, "f_distance"] = dists
            split.loc[unique_indices, "f_cluster"] = clusters
            split.loc[unique_indices, "f_fill"] = fills
            split.interpolate(method="cubic", inplace=True)
            full_features = split[["f_distance", "f_cluster", "f_fill"]].to_numpy()
            with feature_path.open("wb") as file:
                np.save(file, full_features)
