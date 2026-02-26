"""Contains the abstract `Algorithm` class and concrete algorithm implementations.

This package provides a standardized framework for defining and evaluating
different eye movement event classification algorithms, such as IV-T, ID-T,
and others.
"""

from .algorithm import Algorithm, AlgorithmParams
from .idt import IDT, IDTParams
from .ivt import IVT, IVTParams
from .steil import Steil, SteilParams
from .two_threshold import TwoThreshold, TwoThresholdParams
from .with_sp_classifier import WithSPClassifier

__all__ = [
    "IDT",
    "IVT",
    "Algorithm",
    "AlgorithmParams",
    "IDTParams",
    "IVTParams",
    "Steil",
    "SteilParams",
    "TwoThreshold",
    "TwoThresholdParams",
    "WithSPClassifier",
]
