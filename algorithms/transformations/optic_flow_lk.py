"""Provides a transformation for calculating optical flow via the Lucas-Kanade method."""

from itertools import product

import cv2
import numpy as np
from gaze_dataset import GazeDataset
from tqdm import tqdm

from constants import LK

from .transformation import Transformation


class OpticFlowLK(Transformation):
    """Calculates sparse optical flow using the Lucas-Kanade (LK) algorithm.

    This transformation iterates through the video frames of each data split,
    calculates the optical flow between consecutive frames using
    `cv2.calcOpticalFlowPyrLK` on a predefined grid of points, and then adds
    the mean of the calculated flow vectors to the DataFrame.
    """

    def apply(self, data: GazeDataset):
        """Apply the LK optical flow calculation to all splits in the dataset.

        Args:
            data: The `GazeDataset` to be transformed.

        """
        for params, split in tqdm(data.get_splits()):
            video_path = data.get_video_path(params)
            video = cv2.VideoCapture(video_path)
            frames = []
            success, frame = video.read()

            while success:
                frames.append(frame)
                success, frame = video.read()
            video.release()

            flows = []

            for idx in range(len(frames) - 1):
                first, second = frames[idx], frames[idx + 1]

                first = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
                second = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)

                #     # Note: changing from rounding to nearest int to float does not significantly improve performance
                x_coords = np.linspace(LK.OFFSET, data.width - LK.OFFSET, num=11, dtype=int)
                y_coords = np.linspace(LK.OFFSET, data.height - LK.OFFSET, num=11, dtype=int)
                #     # x_coords = np.linspace(LK_OFFSET, WIDTH - LK_OFFSET, num=11)
                #     # y_coords = np.linspace(LK_OFFSET, HEIGHT - LK_OFFSET, num=11)
                grid = np.array(list(product(x_coords, y_coords)), dtype=np.float32).reshape(
                    -1,
                    1,
                    2,
                )
                flow = cv2.calcOpticalFlowPyrLK(
                    first,
                    second,
                    grid,
                    None,
                    winSize=(LK.WIN_SIZE, LK.WIN_SIZE),
                    maxLevel=LK.MAX_LEVEL,
                )
                new_points, valid, _ = flow
                valid = np.repeat(valid[:, :, np.newaxis], 2, axis=2)
                new_points = np.where(valid == 1, new_points, np.nan).reshape(11, 11, 2)
                flow = new_points - grid.reshape(11, 11, 2)
                flows.append(flow)

            flows = np.stack(flows, axis=0)

            average_flow = np.nanmean(flows.astype(np.float64), axis=(1, 2))
            # TODO: this might need some logic checks
            split["frame"] = split["frame"].apply(lambda x: np.nan if x == 0 else int(x))
            split.dropna(inplace=True)
            indices = split["frame"].astype(int).to_numpy() - 1
            split[["flow_x", "flow_y"]] = average_flow[indices]
