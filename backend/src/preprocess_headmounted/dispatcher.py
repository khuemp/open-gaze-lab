"""Auto-detect dispatcher between the DD and GiW head-mounted loaders.

The rule is intentionally extension-only: any ``.mat`` entry inside the
uploaded ZIP routes to the GiW loader, otherwise the DD loader runs. Each
loader still validates its own required files and raises a clear error if
anything is missing.
"""

import zipfile

from .dd import load_npy_dataset
from .giw import load_giw_dataset


def load_head_mounted_dataset(zip_path: str, sampling_rate_hz: float,
                              video_path: str | None = None):
    """Pick the right loader by peeking at the ZIP contents.

    Args:
        zip_path: Path to the uploaded dataset ZIP.
        sampling_rate_hz: Gaze sampling rate in Hz, supplied by the caller.
        video_path: Path to the scene-camera video. Required for GiW (used to
            derive frame timestamps from the video FPS); unused but accepted
            by the DD loader for signature symmetry.
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()

    if _has_mat_entry(names):
        if video_path is None:
            raise ValueError(
                "GiW datasets require the scene video to derive frame timestamps"
            )
        return load_giw_dataset(zip_path, sampling_rate_hz=sampling_rate_hz,
                                video_path=video_path)
    return load_npy_dataset(zip_path, sampling_rate_hz=sampling_rate_hz,
                            video_path=video_path)


def _has_mat_entry(names) -> bool:
    """Return True if any non-resource-fork entry has a ``.mat`` basename."""
    for name in names:
        basename = name.replace("\\", "/").rsplit("/", 1)[-1]
        if basename.startswith("._"):
            continue
        if basename.lower().endswith(".mat"):
            return True
    return False
