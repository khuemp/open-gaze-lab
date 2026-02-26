"""Implements the event classification algorithm from Steil et al. (2018)."""

from dataclasses import dataclass

import numpy as np
from transformations import (
    SavgolFilter,
    Transformation,
)

from algorithms import Algorithm, AlgorithmParams
from datasets import GazeDataset
from event_matching.event import EventType


@dataclass(frozen=True)
class SteilParams(AlgorithmParams):
    """Dataclass for the Steil et al. (2018) algorithm parameters.

    Attributes:
        threshold: The patch distance threshold for classifying fixations.

    """

    threshold: int = 100


class Steil(Algorithm[SteilParams]):
    """An implementation of the algorithm from Steil et al. (2018).

    This algorithm classifies gaze samples into fixations or saccades based on
    the visual similarity of image patches around consecutive gaze points. A
    distance above the threshold is classified as a saccade, and below as a
    fixation.
    """

    def __init__(self, data: GazeDataset, *, load_original: bool):
        """Initialise the algorithm.

        Args:
            data: The `GazeDataset` to be processed.
            load_original: A flag to determine whether to load original,
                           pre-computed patch similarities from Drews and
                           Dierkes or predict them ourselves.
        """
        preprocessing: list[Transformation] = [
            SavgolFilter(55, data.sample_rate_hz),
            # PatchSimilarityLoaderDrews() if load_original else PatchSimilarityPredictor(),
        ]

        super().__init__(data, preprocessing)

    def classify(self, params: SteilParams) -> None:
        """Classify events based on the patch similarity distance threshold.

        Args:
            params: The parameters for the classification.

        """
        self.matchers = []

        for _key, split in self.data.get_splits():
            split["prediction"] = np.where(
                split["patch_dist"] > params.threshold,
                EventType.FIXATION,
                EventType.SACCADE,
            )
