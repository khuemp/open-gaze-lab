"""Utilities for loading head-mounted eye-tracking datasets from .npy files.

Supports the Drews dataset format used by Pupil Invisible glasses recordings:
  - gaze.npy          (N, 2)  — eye position x, y in pixels
  - time_gaze.npy     (N,)    — gaze timestamps in seconds
  - optic_flow.npy    (M, 11, 11, 2) — per-frame optical flow grid
  - time_optic_flow.npy (M,)  — video frame timestamps in seconds
  - time_scene_camera.npy (M,) — scene camera frame timestamps in seconds
  - gt_labels.npy     (N,)    — optional ground truth (1=Fixation, 0=Saccade)
"""

import os
import zipfile
import tempfile
import shutil

import numpy as np
import pandas as pd

# Files we look for inside the dataset ZIP. Anything else in the archive is ignored.
_REQUIRED_NPY_FILES = (
    "gaze.npy",
    "time_gaze.npy",
    "optic_flow.npy",
    "time_optic_flow.npy",
    "time_scene_camera.npy",
)
_OPTIONAL_NPY_FILES = ("gt_labels.npy",)


def load_npy_dataset(zip_path: str, sampling_rate_hz: float):
    """Load a head-mounted eye-tracking dataset from a ZIP of .npy files.

    Args:
        zip_path: Path to the ZIP file containing .npy files.
        sampling_rate_hz: Gaze sampling rate in Hz, supplied by the caller.
            The loader does not infer it from the data.

    Returns:
        Tuple of (DataFrame, metadata_dict).
        DataFrame columns: x, y, timestamp, flow_x, flow_y, video_timestamp, frame
            (and optionally gt_label).
        Timestamps are in **seconds** (EventDetection.__init__ auto-converts to ms).
        metadata_dict keys: video_start_time, sampling_rate_hz, has_gt_labels,
            n_gaze_samples, n_video_frames.
    """
    tmp_dir = tempfile.mkdtemp(prefix="eyetrack_npy_")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            extracted = _extract_known_npys(zf, tmp_dir)

        if "gaze.npy" not in extracted:
            raise FileNotFoundError(
                "ZIP must contain gaze.npy (at root or in a single subfolder). "
                f"Found relevant files: {sorted(extracted)}"
            )

        gaze = np.load(extracted["gaze.npy"])                     # (N, 2)
        time_gaze = np.load(extracted["time_gaze.npy"])           # (N,)
        optic_flow = np.load(extracted["optic_flow.npy"])         # (M, 11, 11, 2)
        time_of = np.load(extracted["time_optic_flow.npy"])       # (M,)
        time_sc = np.load(extracted["time_scene_camera.npy"])     # (M,)

        gt_labels = np.load(extracted["gt_labels.npy"]) if "gt_labels.npy" in extracted else None

        df = _build_dataframe(gaze, time_gaze, optic_flow, time_of, time_sc, gt_labels)

        metadata = {
            "video_start_time": float(time_sc[0]),
            "sampling_rate_hz": float(sampling_rate_hz),
            "has_gt_labels": gt_labels is not None,
            "n_gaze_samples": len(time_gaze),
            "n_video_frames": len(time_sc),
        }

        return df, metadata
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def load_npy_dataset_from_dir(data_dir: str, sampling_rate_hz: float):
    """Load a head-mounted dataset directly from a directory of .npy files.

    Same as load_npy_dataset but without the ZIP extraction step.
    Useful for testing with on-disk datasets.
    """
    gaze = np.load(os.path.join(data_dir, "gaze.npy"))
    time_gaze = np.load(os.path.join(data_dir, "time_gaze.npy"))
    optic_flow = np.load(os.path.join(data_dir, "optic_flow.npy"))
    time_of = np.load(os.path.join(data_dir, "time_optic_flow.npy"))
    time_sc = np.load(os.path.join(data_dir, "time_scene_camera.npy"))

    gt_path = os.path.join(data_dir, "gt_labels.npy")
    gt_labels = np.load(gt_path) if os.path.exists(gt_path) else None

    df = _build_dataframe(gaze, time_gaze, optic_flow, time_of, time_sc, gt_labels)

    metadata = {
        "video_start_time": float(time_sc[0]),
        "sampling_rate_hz": float(sampling_rate_hz),
        "has_gt_labels": gt_labels is not None,
        "n_gaze_samples": len(time_gaze),
        "n_video_frames": len(time_sc),
    }

    return df, metadata


def extract_video_metadata(video_path: str) -> dict:
    """Extract metadata from a video file using OpenCV.

    Returns:
        Dict with keys: fps, width, height, duration_s, n_frames.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_s = n_frames / fps if fps > 0 else 0.0
        return {
            "fps": fps,
            "width": width,
            "height": height,
            "duration_s": duration_s,
            "n_frames": n_frames,
        }
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_known_npys(zf: zipfile.ZipFile, dest_dir: str) -> dict:
    """Extract only the known dataset .npy files from *zf* into *dest_dir*.

    Anything else in the archive (videos, JSON, extra arrays, nested folders)
    is ignored. Matches by basename, so files may live at the root or any
    subfolder inside the ZIP. Returns a mapping basename -> extracted path.
    """
    wanted = set(_REQUIRED_NPY_FILES) | set(_OPTIONAL_NPY_FILES)
    extracted: dict = {}

    for info in zf.infolist():
        if info.is_dir():
            continue
        # ZIP entries always use '/' separators per the spec.
        basename = info.filename.replace("\\", "/").rsplit("/", 1)[-1]
        # Skip macOS resource-fork files like '._gaze.npy' and any non-target file.
        if basename.startswith("._") or basename not in wanted or basename in extracted:
            continue
        out_path = os.path.join(dest_dir, basename)
        with zf.open(info) as src, open(out_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
        extracted[basename] = out_path

    return extracted


def _build_dataframe(gaze, time_gaze, optic_flow, time_of, time_sc, gt_labels=None):
    """Build a gaze DataFrame with per-sample optic flow mapped from video frames.

    Uses np.searchsorted to map each gaze sample to the nearest preceding video
    frame — matching the pattern from verify_comparison.py prepare_dataframe().
    """
    # Average the 11×11 flow grid to a single vector per frame
    mean_flow = np.nanmean(optic_flow.astype(np.float64), axis=(1, 2))  # (M, 2)

    # Map each gaze timestamp to the nearest preceding video frame
    indices = np.searchsorted(time_of, time_gaze, side="right") - 1
    indices = np.clip(indices, 0, len(time_of) - 1)

    flow_per_gaze = mean_flow[indices]          # (N, 2)
    video_ts_per_gaze = time_of[indices]        # (N,)

    # Also compute frame index via scene_camera timestamps
    frame_indices = np.searchsorted(time_sc, time_gaze, side="right") - 1
    frame_indices = np.clip(frame_indices, 0, len(time_sc) - 1)

    data = {
        "timestamp": time_gaze,
        "x": gaze[:, 0],
        "y": gaze[:, 1],
        "flow_x": flow_per_gaze[:, 0],
        "flow_y": flow_per_gaze[:, 1],
        "video_timestamp": video_ts_per_gaze,
        "frame": frame_indices,
    }

    if gt_labels is not None:
        data["gt_label"] = gt_labels

    return pd.DataFrame(data)
