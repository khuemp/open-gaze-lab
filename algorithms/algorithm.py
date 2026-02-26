"""Defines the abstract base class for event classification algorithms.

This module provides the `Algorithm` ABC, which serves as a standardized
framework for implementing, running, and evaluating different eye movement
classification algorithms.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from transformations import Transformation

from datasets import GazeDataset
from event_matching import INT_TO_EVENT_NAME, EventType, IOUMatcher


@dataclass(frozen=True)
class AlgorithmParams:
    """A base dataclass for algorithm-specific parameters."""

    pass


T_AlgorithmParams = TypeVar("T_AlgorithmParams", bound=AlgorithmParams)


class Algorithm(Generic[T_AlgorithmParams], ABC):
    """An abstract base class for an event classification algorithm.

    This class provides a common structure for applying preprocessing
    transformations to a dataset, running a classification task, and scoring
    the results against ground truth labels.

    Attributes:
        data: The GazeDataset instance to be processed.
        preprocessing: A list of `Transformation` objects to be applied to the
                       data before classification.
        matchers: A list of `IOUMatcher` instances used for event-based scoring.
                  Every recording of a dataset needs its own matcher.

    """

    def __init__(self, data: GazeDataset, preprocessing: list[Transformation]):
        """Initialise the algorithm with a dataset and preprocessing steps.

        Args:
            data: The `GazeDataset` to be processed.
            preprocessing: A list of `Transformation` instances to be applied
                           sequentially to the dataset.

        """
        self.data = data
        self.preprocessing = preprocessing
        self.matchers: list[IOUMatcher] = []
        self._apply_preprocessing()

    def _assert_column_present(self, col_name: str):
        """Check if a column exists in all data splits."""
        for _, split in self.data.get_splits():
            assert col_name in split.columns

    def _apply_preprocessing(self):
        """Apply all preprocessing transformations to the dataset."""
        for transform in self.preprocessing:
            transform.apply(self.data)

    def get_ground_truth(self) -> pd.Series:
        """Concatenate the ground truth labels from all data splits.

        Returns:
            A pandas Series containing all ground truth labels.

        """
        self._assert_column_present("label")
        return pd.concat(
            [split["label"] for _, split in self.data.get_splits()],
            ignore_index=True,
            axis=0,
        )

    def get_prediction(self) -> pd.Series:
        """Concatenate the prediction labels from all data splits.

        Should only be run after classification has taken place.

        Returns:
            A pandas Series containing all predicted labels.

        """
        self._assert_column_present("prediction")
        return pd.concat(
            [split["prediction"] for _, split in self.data.get_splits()],
            ignore_index=True,
            axis=0,
        )

    def get_sample_f1_score(self, event_type: EventType) -> float:
        """Calculate the sample-wise F1 score for a specific event type.

        Args:
            event_type: The event type to score.

        Returns:
            The F1 score for the specified event type.

        """
        ground_truth = self.get_ground_truth()
        prediction = self.get_prediction()
        # return cohen_kappa_score(ground_truth, prediction)
        return f1_score(
            ground_truth,
            prediction,
            average=None,
            labels=[event_type],
        )

    def print_sample_classsification_report(self):
        """Print a detailed sample-wise classification report."""
        labels = np.unique(self.get_ground_truth())
        target_names = list(map(lambda x: INT_TO_EVENT_NAME(x), labels))
        print(
            classification_report(
                self.get_ground_truth(),
                self.get_prediction(),
                target_names=target_names,
            ),
        )

    @abstractmethod
    def classify(self, params: T_AlgorithmParams) -> None:
        """Run the event classification algorithm with the given parameters."""
        ...

    def score(self, dist: float, slen: int, flen: int, *, return_fixation: bool) -> float:
        """Calculate the event-based F1 score using an IoU matcher.

        This method first performs event matching between ground truth and
        predictions for each data split. It then calculates a classification
        report and returns either the F1 score for fixations or the weighted
        average F1 score.

        Args:
            dist: The minimum angular distance for filtering micro-saccades.
            slen: The minimum duration for filtering micro-saccades.
            flen: The minimum duration for filtering short fixations.
            return_fixation: If True, returns the F1 score for fixations;
                             otherwise, returns the weighted average F1 score.

        Returns:
            The calculated F1 score.
        """
        if len(self.matchers) == 0:
            for _key, split in self.data.get_splits():
                self.matchers.append(IOUMatcher(split))
                # self.matchers.append(MaximumOverlapMatcher(split))

        assert len(self.matchers) == len(self.data.get_splits())

        scores = []
        for matcher in self.matchers:
            matches = matcher.match(dist, slen, flen)

            def mapping(x):
                if x is None:
                    return "None"
                return INT_TO_EVENT_NAME[x.label]

            gt = [mapping(x[0]) for x in matches]
            pred = [mapping(x[1]) for x in matches]

            report = classification_report(
                gt,
                pred,
                zero_division=0,
                digits=4,
                output_dict=True,
                labels=["Fixation", "Following", "Saccade", "Pursuit"],
            )
            scores.append(
                report["Fixation"]["f1-score"]
                if return_fixation
                else report["weighted avg"]["f1-score"],
            )

        return sum(scores) / len(scores)
