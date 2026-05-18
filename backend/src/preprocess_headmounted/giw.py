"""Head-mounted eye-tracker preprocessing for the Gaze-in-Wild (GiW) dataset.

Expected files inside the dataset ZIP (matched by filename, so they may sit
at the archive root or in any subfolder):

* ``PrIdx_<P>_TrIdx_<T>.mat``        signals (gaze + timestamps + frame indices)
* ``PrIdx_<P>_TrIdx_<T>_Lbr_<N>.mat`` one or more labeler annotation files
* ``optic_flow.npy`` ``(M, 2)`` per-frame mean flow
"""

import os
import re
import shutil
import tempfile
import zipfile

import numpy as np
import pandas as pd
from scipy.io import loadmat

from .common import extract_video_metadata


# GiW label-class integers that count as "stable gaze" for the binary
# collapse: FIXATION (1) and FOLLOWING (5). Following = head+eye co-rotation
# tracking an attended target; in real-world recordings strict fixation alone
# is rare (~1% of samples), so we lump it with following to match what the
# OpenGazeLab F1 scoring treats as Fixation. Saccade, pursuit, blink, and
# undefined all collapse to 0.
_STABLE_GAZE_VALUES = (1, 5)

_GIW_DEFAULT_WIDTH = 1920
_GIW_DEFAULT_HEIGHT = 1080

_SIGNALS_RE = re.compile(r"PrIdx_(\d+)_TrIdx_(\d+)\.mat$", re.IGNORECASE)
_LABEL_RE = re.compile(r"_Lbr_(\d+)\.mat$", re.IGNORECASE)


def load_giw_dataset(zip_path: str, sampling_rate_hz: float, video_path: str):
    """Load a GiW head-mounted dataset from a ZIP of .mat + .npy files.

    Returns ``(DataFrame, metadata_dict)`` matching the contract of
    :func:`preprocess_headmounted.dd.load_npy_dataset`, so the rest of the
    pipeline does not need to special-case GiW.
    """
    tmp_dir = tempfile.mkdtemp(prefix="eyetrack_giw_")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            extracted = _extract_giw_files(zf, tmp_dir)

        signals_path = extracted["signals"]
        label_paths = extracted["labels"]
        flow_path = extracted["flow"]

        participant_id, trial_id = _parse_signals_basename(os.path.basename(signals_path))

        gaze, timestamps, frames = _load_signals(signals_path)
        labels_binary = _load_and_collapse_labels(label_paths, trial_id)
        optic_flow = _load_flow(flow_path)

        video_meta = extract_video_metadata(video_path)
        fps = float(video_meta.get("fps") or 0.0)
        if fps <= 0.0:
            raise ValueError(f"Could not read FPS from video file: {video_path}")
        n_video_frames = int(video_meta.get("n_frames") or 0)

        df = _build_dataframe(
            gaze=gaze,
            timestamps=timestamps,
            frames=frames,
            labels_binary=labels_binary,
            optic_flow=optic_flow,
            fps=fps,
        )

        metadata = {
            "video_start_time": 0.0,
            "sampling_rate_hz": float(sampling_rate_hz),
            "has_gt_labels": True,
            "n_gaze_samples": int(len(timestamps)),
            "n_video_frames": int(n_video_frames or len(optic_flow)),
            "participant_id": participant_id,
            "trial_id": trial_id,
        }
        return df, metadata
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_giw_files(zf: zipfile.ZipFile, dest_dir: str) -> dict:
    """Pull GiW files out of the ZIP. Returns dict with ``signals``,
    ``labels`` (list), and ``flow`` paths."""
    signals_path = None
    label_paths: list = []
    flow_path = None

    for info in zf.infolist():
        if info.is_dir():
            continue
        basename = info.filename.replace("\\", "/").rsplit("/", 1)[-1]
        if basename.startswith("._"):
            continue
        lower = basename.lower()

        is_label = bool(_LABEL_RE.search(basename))
        is_signals = lower.endswith(".mat") and not is_label
        is_flow = lower.endswith(".npy") and (
            lower == "optic_flow.npy"
        )

        if not (is_label or is_signals or is_flow):
            continue

        out_path = os.path.join(dest_dir, basename)
        with zf.open(info) as src, open(out_path, "wb") as dst:
            shutil.copyfileobj(src, dst)

        if is_signals:
            if signals_path is not None:
                raise ValueError(
                    "GiW ZIP must contain exactly one signals .mat file "
                    "(no _Lbr_ suffix); found multiple."
                )
            signals_path = out_path
        elif is_label:
            label_paths.append(out_path)
        elif is_flow:
            if flow_path is not None:
                raise ValueError(
                    "GiW ZIP must contain exactly one optic_flow.npy "
                )
            flow_path = out_path

    if signals_path is None:
        raise FileNotFoundError(
            "GiW ZIP must contain a signals .mat file matching "
            "PrIdx_<P>_TrIdx_<T>.mat."
        )
    if not label_paths:
        raise FileNotFoundError(
            "GiW ZIP must contain at least one labeler .mat file matching "
            "PrIdx_<P>_TrIdx_<T>_Lbr_<N>.mat."
        )
    if flow_path is None:
        raise FileNotFoundError(
            "GiW ZIP must contain optic_flow.npy with shape (M, 2)."
        )

    return {"signals": signals_path, "labels": label_paths, "flow": flow_path}


def _parse_signals_basename(basename: str) -> tuple[int | None, int | None]:
    match = _SIGNALS_RE.search(basename)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def _load_signals(signals_path: str):
    """Read gaze (POR), timestamps (T), and per-sample frame indices."""
    data = loadmat(signals_path, simplify_cells=True)
    process = data["ProcessData"]

    por = np.asarray(process["ETG"]["POR"], dtype=np.float64)
    if por.ndim != 2 or por.shape[1] != 2:
        raise ValueError(f"Expected ProcessData.ETG.POR with shape (N, 2); got {por.shape}")
    gaze = np.clip(por, 0.0, 1.0) * np.array([_GIW_DEFAULT_WIDTH, _GIW_DEFAULT_HEIGHT])

    timestamps = np.asarray(process["T"], dtype=np.float64).reshape(-1)

    # Frame indices live in a nested cell-array slot that simplify_cells
    # collapses incorrectly, so reload raw and index the struct directly.
    raw = loadmat(signals_path)
    frames = np.asarray(raw["ProcessData"]["ETG"][0, 0][0, 0][5][0], dtype=np.int64).reshape(-1)

    n = min(len(gaze), len(timestamps), len(frames))
    return gaze[:n], timestamps[:n], frames[:n]


def _load_and_collapse_labels(label_paths: list[str], trial_id: int | None) -> np.ndarray:
    """Apply priority rule, then collapse FIXATION ∪ FOLLOWING → 1, else → 0."""
    by_id: dict[int, np.ndarray] = {}
    for path in label_paths:
        basename = os.path.basename(path)
        match = _LABEL_RE.search(basename)
        if not match:
            continue
        labeler_id = int(match.group(1))
        raw = loadmat(path, simplify_cells=True)["LabelData"]["Labels"]
        by_id[labeler_id] = np.asarray(raw, dtype=np.int64).reshape(-1)

    if not by_id:
        raise ValueError("No labeler files matched _Lbr_<N>.mat naming.")

    chosen_id = _pick_labeler(by_id, trial_id)
    labels = by_id[chosen_id]
    return np.isin(labels, _STABLE_GAZE_VALUES).astype(np.int64)


def _pick_labeler(by_id: dict[int, np.ndarray], trial_id: int | None) -> int:
    """Replicate giw_dataset.py:182-206 priority order, with a sorted fallback."""
    if trial_id == 1 and 5 in by_id:
        return 5
    for preferred in (6, 5, 1, 2, 3):
        if preferred in by_id:
            return preferred
    return min(by_id.keys())


def _load_flow(flow_path: str) -> np.ndarray:
    """Load the (M, 2) per-frame mean optical flow array."""
    flow = np.load(flow_path)
    if flow.ndim != 2 or flow.shape[1] != 2:
        raise ValueError(
            f"Expected optic_flow.npy with shape (M, 2); got {flow.shape}. "
            "Average the per-frame flow grid before packaging it."
        )
    return flow.astype(np.float64)


def _build_dataframe(*, gaze, timestamps, frames, labels_binary, optic_flow, fps):
    n = min(len(gaze), len(timestamps), len(frames), len(labels_binary))
    gaze = gaze[:n]
    timestamps = timestamps[:n]
    frames = frames[:n]
    labels_binary = labels_binary[:n]

    frames = np.clip(frames, 0, len(optic_flow) - 1).astype(np.int64)
    flow_xy = optic_flow[frames]

    return pd.DataFrame({
        "timestamp": timestamps,
        "x": gaze[:, 0],
        "y": gaze[:, 1],
        "flow_x": flow_xy[:, 0],
        "flow_y": flow_xy[:, 1],
        "video_timestamp": frames.astype(np.float64) / fps,
        "frame": frames,
        "gt_label": labels_binary,
    })
