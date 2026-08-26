"""Head-mounted eye-tracker preprocessing for the Drews & Dierkes (DD) dataset.

Expected files inside the dataset ZIP (matched by basename, so they may
sit at the archive root or in any single subfolder):

* ``gaze.npy``            ``(N, 2)``   eye position x, y in pixels
* ``time_gaze.npy``       ``(N,)``     gaze timestamps in seconds
* ``optic_flow.npy``      ``(M, 11, 11, 2)`` per-frame 11x11 flow grid
* ``time_optic_flow.npy`` ``(M,)``     video frame timestamps in seconds
* ``time_scene_camera.npy`` ``(M,)``   scene-camera frame timestamps in seconds
* ``gt_labels.npy``       ``(N,)``     optional ground truth (1=Fixation, 0=Saccade)
"""

import os
import shutil
import tempfile
import zipfile

import numpy as np
import pandas as pd


_REQUIRED_NPY_FILES = (
    "gaze.npy",
    "time_gaze.npy",
    "optic_flow.npy",
    "time_optic_flow.npy",
    "time_scene_camera.npy",
)
_OPTIONAL_NPY_FILES = ("gt_labels.npy",)


def load_npy_dataset(zip_path: str, sampling_rate_hz: float, video_path: str | None = None):
    """Load a head-mounted eye-tracking dataset from a ZIP of .npy files.

    Args:
        zip_path: Path to the ZIP file containing .npy files.
        sampling_rate_hz: Gaze sampling rate in Hz, supplied by the caller.
            The loader does not infer it from the data.
        video_path: Unused by the DD loader; accepted for signature symmetry
            with :func:`load_giw_dataset` so the dispatcher can call both
            uniformly.

    Returns:
        Tuple ``(DataFrame, metadata_dict)``.
        DataFrame columns: ``x``, ``y``, ``timestamp``, ``flow_x``, ``flow_y``,
        ``video_timestamp``, ``frame`` (and optionally ``gt_label``).
        Timestamps are in **seconds** — ``EventDetection.__init__`` converts
        them to milliseconds.
        ``metadata_dict`` keys: ``video_start_time``, ``sampling_rate_hz``,
        ``has_gt_labels``, ``n_gaze_samples``, ``n_video_frames``.
    """
    del video_path  # signature symmetry only — DD has its own scene-camera timestamps

    tmp_dir = tempfile.mkdtemp(prefix="eyetrack_npy_")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            extracted = _extract_known_npys(zf, tmp_dir)

        if "gaze.npy" not in extracted:
            raise FileNotFoundError(
                "ZIP must contain gaze.npy (at root or in a single subfolder). "
                f"Found relevant files: {sorted(extracted)}"
            )

        return _load_from_paths(extracted, sampling_rate_hz)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_from_paths(paths: dict, sampling_rate_hz: float):
    """Load arrays from *paths* (basename -> filepath) and build the DataFrame."""
    gaze       = np.load(paths["gaze.npy"])             # (N, 2)
    time_gaze  = np.load(paths["time_gaze.npy"])        # (N,)
    optic_flow = np.load(paths["optic_flow.npy"])       # (M, 11, 11, 2)
    time_of    = np.load(paths["time_optic_flow.npy"])  # (M,)
    time_sc    = np.load(paths["time_scene_camera.npy"])  # (M,)
    gt_labels  = np.load(paths["gt_labels.npy"]) if "gt_labels.npy" in paths else None

    df = _build_dataframe(gaze, time_gaze, optic_flow, time_of, time_sc, gt_labels)

    metadata = {
        "video_start_time": float(time_sc[0]),
        "sampling_rate_hz": float(sampling_rate_hz),
        "has_gt_labels": gt_labels is not None,
        "n_gaze_samples": len(time_gaze),
        "n_video_frames": len(time_sc),
    }
    return df, metadata


def _extract_known_npys(zf: zipfile.ZipFile, dest_dir: str) -> dict:
    """Extract only the dataset .npy files we care about from *zf*.

    Matches by basename, so files may live at the root or any subfolder
    inside the ZIP. macOS resource-fork files (``._gaze.npy``) and any
    unrelated entries are skipped. Returns ``{basename: extracted_path}``.
    """
    wanted = set(_REQUIRED_NPY_FILES) | set(_OPTIONAL_NPY_FILES)
    extracted: dict = {}

    for info in zf.infolist():
        if info.is_dir():
            continue
        # ZIP entries always use '/' separators per the spec.
        basename = info.filename.replace("\\", "/").rsplit("/", 1)[-1]
        if basename.startswith("._") or basename not in wanted or basename in extracted:
            continue
        out_path = os.path.join(dest_dir, basename)
        with zf.open(info) as src, open(out_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
        extracted[basename] = out_path

    return extracted


def _build_dataframe(gaze, time_gaze, optic_flow, time_of, time_sc, gt_labels=None):
    """Build a per-gaze-sample DataFrame with optic flow mapped from video frames.

    Each gaze sample is mapped to the nearest preceding video frame via
    ``np.searchsorted`` — matching the pattern from the reference research code.
    """
    # Average the 11x11 flow grid to a single (fx, fy) per frame.
    mean_flow = np.nanmean(optic_flow.astype(np.float64), axis=(1, 2))  # (M, 2)

    # Map each gaze timestamp to the nearest preceding video frame.
    flow_idx = np.clip(np.searchsorted(time_of, time_gaze, side="right") - 1,
                       0, len(time_of) - 1)
    frame_idx = np.clip(np.searchsorted(time_sc, time_gaze, side="right") - 1,
                        0, len(time_sc) - 1)

    data = {
        "timestamp": time_gaze,
        "x": gaze[:, 0],
        "y": gaze[:, 1],
        "flow_x": mean_flow[flow_idx, 0],
        "flow_y": mean_flow[flow_idx, 1],
        "video_timestamp": time_of[flow_idx],
        "frame": frame_idx,
    }
    if gt_labels is not None:
        data["gt_label"] = gt_labels

    return pd.DataFrame(data)
