"""Implements an I-VT algorithm augmented with a Saccade/Pursuit classifier."""

import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from transformations import (
    AdaptiveThreshold,
    RelativeVelocityCalculator,
    SavgolFilter,
    Transformation,
)

from algorithms import Algorithm, IVTParams
from constants import PROJECT_ROOT
from datasets import GazeDataset
from event_matching.event import EventType


class WithSPClassifier(Algorithm[IVTParams]):
    """An I-VT algorithm that uses a classifier to distinguish Saccade/Pursuit.

    This algorithm first uses a  trained logistic regression classifier to
    distinguish between non-pursuit and smooth pursuit. It then uses a
    velocity threshold to separate fixations from saccades for the remaining samples.
    """

    def __init__(self, data: GazeDataset, *, feat_col: str, fold_num: int):
        """Initialise the algorithm and its preprocessing steps.

        Args:
            data: The `GazeDataset` to be processed.
            feat_col: The name of the additional feature column to be used by
                      the logistic regression classifier.
            fold_num: The cross-validation fold number, used for loading the
                      correct pre-trained model.

        """
        self.data = data
        self.feat_col = feat_col
        self.classifier = None
        self.feat_col = feat_col
        self.model_path = Path(
            PROJECT_ROOT / f"models/{data.name}/lr_fold_{fold_num}_{self.feat_col}.pkl",
        )
        preprocessing: list[Transformation] = [
            SavgolFilter(55, data.sample_rate_hz),
            RelativeVelocityCalculator(x_col="filter_x", y_col="filter_y"),
        ]
        super().__init__(data, preprocessing)

    def train(self, data: GazeDataset):
        """Train the logistic regression classifier for Saccade vs. Pursuit.

        Args:
            data: The dataset to use for training.

        """
        self.classifier = LogisticRegression(class_weight="balanced")
        features = []
        labels = []
        for _, split in data.get_splits():
            features.append(split[["vel_rel_mag", self.feat_col]])
            labels.append(split["label"])

        X = np.nan_to_num(np.vstack(features))
        X = X / X.max(axis=0)
        y = np.hstack(labels)
        label_map = np.vectorize(lambda x: 1 if x == EventType.PURSUIT else 0)
        y = label_map(y)
        self.classifier.fit(X, y)

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        with self.model_path.open("wb") as f:
            pickle.dump(self.classifier, f)

    def load_classifier(self):
        """Load the pre-trained Pursuit classifier from a file."""
        with self.model_path.open("rb") as f:
            self.classifier = pickle.load(f)

    def classify(self, params: IVTParams) -> None:
        """Classify events using a threshold and a secondary classifier.

        First, a classifier is used to identify pursuit samples.
        Then, for all remaing samples, a velocity threshold (fixed or adaptive)
        is used to identify fixations.

        Args:
            params: The parameters for the initial I-VT classification.

        """
        self.matchers = []
        if params.gain is None:
            assert params.window_size_ms is None
            for _key, split in self.data.get_splits():
                split["prediction"] = np.where(
                    split["vel_rel_mag"] < params.threshold,
                    EventType.FIXATION,
                    EventType.SACCADE,
                )
                features = split[["vel_rel_mag", self.feat_col]].to_numpy()
                features = np.nan_to_num(features)
                features = features / features.max(axis=0)
                sp_classificiation = self.classifier.predict(features)
                sp_indices = np.where(sp_classificiation == 1)[0]
                split.loc[sp_indices, "prediction"] = EventType.PURSUIT
        else:
            assert params.gain is not None
            assert params.window_size_ms is not None
            assert params.window_size_ms > 0
            assert params.gain >= 0

            window_size = int(params.window_size_ms // self.data.sample_duration_ms)
            process = AdaptiveThreshold(window_size, params.gain, params.threshold)
            process.apply(self.data)

            for _key, split in self.data.get_splits():
                split["prediction"] = np.where(
                    split["vel_rel_mag"] < split["threshold"],
                    EventType.FIXATION,
                    EventType.SACCADE,
                )
                features = split[["vel_rel_mag", self.feat_col]].to_numpy()
                features = np.nan_to_num(features)
                features = features / features.max(axis=0)
                sp_classificiation = self.classifier.predict(features)
                sp_indices = np.where(sp_classificiation == 1)[0]
                split.loc[sp_indices, "prediction"] = EventType.PURSUIT
