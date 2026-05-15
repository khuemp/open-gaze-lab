"""Head-mounted eye-tracker preprocessing.

Two dataset layouts are supported and the public dispatcher routes uploads
between them by peeking at the ZIP contents:

* DD (Drews & Dierkes) — ZIP of ``.npy`` arrays. Loader: :func:`load_npy_dataset`.
* GiW (Gaze-in-Wild) — ZIP of ``.mat`` files plus a pre-averaged optical-flow
  ``.npy``. Loader: :func:`load_giw_dataset`.

Both loaders return ``(DataFrame, metadata_dict)`` with identical schemas so
downstream pipeline code stays dataset-agnostic.
"""

from .common import extract_video_metadata
from .dd import load_npy_dataset
from .dispatcher import load_head_mounted_dataset
from .giw import load_giw_dataset

__all__ = [
    "extract_video_metadata",
    "load_npy_dataset",
    "load_giw_dataset",
    "load_head_mounted_dataset",
]
