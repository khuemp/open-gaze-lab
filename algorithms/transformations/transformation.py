"""Defines the abstract base class for all data transformations."""

from abc import ABC, abstractmethod

from datasets import GazeDataset


class Transformation(ABC):
    """An abstract base class for data transformations.

    All transformations are designed to be applied in-place on a
    `GazeDataset` object, typically by adding or modifying columns in its
    underlying DataFrames.
    """

    @abstractmethod
    def apply(self, data: GazeDataset):
        """Apply the transformation to the dataset.

        Args:
            data: The `GazeDataset` to be transformed.
        """
        ...
