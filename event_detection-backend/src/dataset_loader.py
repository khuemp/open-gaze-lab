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


def load_npy_dataset(zip_path: str):
    """Load a head-mounted eye-tracking dataset from a ZIP of .npy files.

    Args:
        zip_path: Path to the ZIP file containing .npy files.

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
        # Extract ZIP
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_dir)

        # Find the directory containing the .npy files (may be nested one level)
        npy_dir = _find_npy_dir(tmp_dir)

        # Load required arrays
        gaze = np.load(os.path.join(npy_dir, "gaze.npy"))                     # (N, 2)
        time_gaze = np.load(os.path.join(npy_dir, "time_gaze.npy"))           # (N,)
        optic_flow = np.load(os.path.join(npy_dir, "optic_flow.npy"))         # (M, 11, 11, 2)
        time_of = np.load(os.path.join(npy_dir, "time_optic_flow.npy"))       # (M,)
        time_sc = np.load(os.path.join(npy_dir, "time_scene_camera.npy"))     # (M,)

        # Optional ground truth labels
        gt_path = os.path.join(npy_dir, "gt_labels.npy")
        gt_labels = np.load(gt_path) if os.path.exists(gt_path) else None

        # Build DataFrame
        df = _build_dataframe(gaze, time_gaze, optic_flow, time_of, time_sc, gt_labels)

        # Metadata
        dt = np.diff(time_gaze)
        sampling_rate_hz = 1.0 / np.median(dt) if len(dt) > 0 else 30.0

        metadata = {
            "video_start_time": float(time_sc[0]),
            "sampling_rate_hz": round(float(sampling_rate_hz), 1),
            "has_gt_labels": gt_labels is not None,
            "n_gaze_samples": len(time_gaze),
            "n_video_frames": len(time_sc),
        }

        return df, metadata
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def load_npy_dataset_from_dir(data_dir: str):
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

    dt = np.diff(time_gaze)
    sampling_rate_hz = 1.0 / np.median(dt) if len(dt) > 0 else 30.0

    metadata = {
        "video_start_time": float(time_sc[0]),
        "sampling_rate_hz": round(float(sampling_rate_hz), 1),
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

def _find_npy_dir(root: str) -> str:
    """Find the directory inside *root* that contains gaze.npy.

    Handles both flat ZIPs (files at root) and single-subfolder ZIPs.
    """
    if os.path.isfile(os.path.join(root, "gaze.npy")):
        return root
    # Check one level deep
    for entry in os.listdir(root):
        candidate = os.path.join(root, entry)
        if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, "gaze.npy")):
            return candidate
    raise FileNotFoundError(
        "ZIP must contain gaze.npy (at root or in a single subfolder). "
        f"Found: {os.listdir(root)}"
    )


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
