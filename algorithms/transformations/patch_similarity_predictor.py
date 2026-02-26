"""Provides a transformation to predict patch similarity using a deep model.

This module defines the `PatchSimilarityPredictor` class, which uses a
pre-trained 2-channel, 2-stream convolutional neural network to predict the
visual similarity between image patches centered at consecutive gaze points.

Parts from this implementation are from the implementation of Steil et al. (2018)
and the original repo from https://github.com/szagoruyko/cvpr15deepcompare
"""

import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchfile
from gaze_dataset import GazeDataset
from torch.autograd import Variable
from tqdm.auto import tqdm

from constants import PATCH_MODEL_PATH, PATCH_SIZE

from .transformation import Transformation


class PatchSimilarityPredictor(Transformation):
    """Calculates patch similarity scores using a pre-trained deep network.

    This transformation iterates through a dataset, extracts image patches
    around consecutive gaze points, and feeds them into a pre-trained PyTorch
    model to get a similarity score (distance). The results are cached to a
    file to avoid re-computation on subsequent runs.
    """

    def __init__(self):
        """Initialise the predictor and load the pre-trained model."""
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        self.model_fn = deepcompare_2ch2stream

        device_str = "cpu"
        if torch.cuda.is_available():
            device_str = "cuda"
            # to prevent opencv from initializing CUDA in workers
            torch.randn(8).cuda()
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        elif torch.mps.is_available():
            device_str = "mps"
        self.device = torch.device(device_str)

        net = torchfile.load(PATCH_MODEL_PATH)
        params = {}
        for j, branch in enumerate(["fovea", "retina"]):
            counter = 0
            for k, layer in enumerate(net.modules[0].modules[j].modules[1].modules):
                if layer.weight is not None and k in {1, 4, 7, 9}:
                    params[f"{branch}.conv{counter}.weight"] = layer.weight
                    params[f"{branch}.conv{counter}.bias"] = layer.bias
                    counter += 1

        counter = 0
        for k, layer in enumerate(net.modules):
            if layer.weight is not None and k in {1, 3}:
                params[f"fc{counter}.weight"] = layer.weight
                params[f"fc{counter}.bias"] = layer.bias
                counter += 1

        self.model_params = {
            k: Variable(torch.from_numpy(v).float().to(self.device)) for k, v in params.items()
        }

    def apply(self, data: GazeDataset):
        """Calculate and add patch similarity scores to the dataset.

        For each split, it checks for a cached result first. If not found, it
        iterates through the video frames, extracts patches, runs the model,
        and saves the resulting distances to a cache file before adding them
        to the DataFrame.

        Args:
            data: The `GazeDataset` to be transformed.

        """
        MODEL_INPUT_SIZE = 64

        for params, split in tqdm(data.get_splits(), desc="Applying patch similarity"):
            sim_path = data.get_similarity_path(params)
            frames, match_indices = np.unique(split["frame"], return_index=True)
            match_gazes = split[["x", "y"]].iloc[match_indices].to_numpy()
            if sim_path.exists():
                distances = np.loadtxt(sim_path)
                split["patch_dist"] = np.nan
                split.loc[match_indices, "patch_dist"] = distances
                # TODO: might need to use this for wrongly cached version
                # split.loc[match_indices, "patch_dist"] = distances[: match_indices.shape[0]]
            else:
                distances = [0]
                video_path = data.get_video_path(params)
                cap = cv2.VideoCapture(str(video_path))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frames[0])
                success, first_frame = cap.read()
                first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
                # first_patch = extract_patch(first_gray, match_gazes[0])
                first_patch = extract_patch(first_gray, match_gazes[0])
                gaze_idx = 1
                while gaze_idx < match_gazes.shape[0]:
                    success, second_frame = cap.read()
                    if not success:
                        break
                    second_gray = cv2.cvtColor(second_frame, cv2.COLOR_BGR2GRAY)
                    # second_patch = extract_patch(second_gray, match_gazes[gaze_idx])
                    second_patch = extract_patch(second_gray, match_gazes[gaze_idx])
                    gaze_idx += 1

                    net_input = np.zeros((2, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
                    if first_patch.size == 0 or second_patch.size == 0:
                        distances.append(0)
                        first_patch = second_patch
                        continue

                    net_input[0, :, :] = cv2.resize(
                        first_patch,
                        (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
                    )
                    net_input[1, :, :] = cv2.resize(
                        second_patch,
                        (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
                    )
                    distance = self.model_fn(
                        torch.from_numpy(net_input).float().to(self.device),
                        self.model_params,
                    )
                    distances.append(distance.item())
                    first_patch = second_patch

                distances = [0] * (len(distances) - match_gazes.shape[0]) + distances

                np.savetxt(sim_path, distances)

                split["patch_dist"] = np.nan
                split.loc[match_indices, "patch_dist"] = distances

            split.interpolate(method="cubic", inplace=True)


def conv2d(input, params, base, stride=1, padding=0):
    """Perform a 2D convolution using pre-loaded parameters."""
    return F.conv2d(input, params[base + ".weight"], params[base + ".bias"], stride, padding)


def linear(input, params, base):
    """Perform a linear transformation using pre-loaded parameters."""
    return F.linear(input, params[base + ".weight"], params[base + ".bias"])


def deepcompare_2ch2stream(input, params):
    """Define the forward pass for the 2-channel, 2-stream siamese network.

    Args:
        input: The input tensor containing two image patches.
        params: A dictionary of pre-loaded model weights and biases.

    Returns:
        The computed distance (similarity score) between the two patches.

    """

    def stream(input, name):
        o = conv2d(input, params, name + ".conv0")
        o = F.max_pool2d(F.relu(o), 2, 2)
        o = conv2d(o, params, name + ".conv1")
        o = F.max_pool2d(F.relu(o), 2, 2)
        o = conv2d(o, params, name + ".conv2")
        o = F.relu(o)
        o = conv2d(o, params, name + ".conv3")
        o = F.relu(o)
        # return o.view(o.size(0), -1)
        return torch.flatten(o)

    o_fovea = stream(F.avg_pool2d(input, 2, 2), "fovea")
    o_retina = stream(F.pad(input, (-16,) * 4), "retina")
    o = linear(torch.cat([o_fovea, o_retina], dim=0), params, "fc0")
    return linear(F.relu(o), params, "fc1")


def extract_patch(frame: np.ndarray, center: tuple[float, float]) -> np.ndarray:
    """Extract a square patch from a frame centered at a given point.

    Args:
        frame: The grayscale source image.
        center: The (x, y) coordinates for the center of the patch.

    Returns:
        The extracted image patch.

    """
    x = center[0]
    y = center[1]
    width = frame.shape[1]
    height = frame.shape[0]
    offset = PATCH_SIZE / 2

    min_x = max(0, round(x - offset))
    max_x = min(width, round(x + offset))
    min_y = max(0, round(y - offset))
    max_y = min(height, round(y + offset))

    return frame[min_y:max_y, min_x:max_x]
