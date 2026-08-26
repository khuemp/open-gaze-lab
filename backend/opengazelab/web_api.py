"""Entry points for running the pipeline inside a browser (Pyodide).

Replaces the former FastAPI layer. Where ``main.py`` parsed multipart form
fields, wrote results under ``data/`` and returned URLs, these functions take
plain bytes and return the results *as strings* — the host page turns them
into blob URLs. Nothing is written to disk except the dataset ZIP, which is
staged in Pyodide's in-memory filesystem so the existing head-mounted loaders
(which take a path) keep working unchanged.

Both functions return::

    {
        "summary":         {...},        # the stat cards
        "events_csv":      "<csv text>", # the downloadable events file
        "plot_html":       "<html>",     # stationary only
        "time_plot_html":  "<html>",     # stationary only
        "video_plot_html": "<html>",     # head-mounted only
    }
"""

import tempfile
from pathlib import Path

from . import EventDetection, EyeTrackingVisualizer
from .preprocess_csv import load_csv_gaze_data
from .preprocess_headmounted import load_head_mounted_dataset, register_video_metadata
from .utils import binary_f1
from .visualization import generate_video_gaze_visualization
from .visualization._image_utils import encode_image_bytes

_VALID_Y_ORIGINS = ("top-left", "top-right", "bottom-left", "bottom-right")


# ---------------------------------------------------------------------------
# Screen-based (CSV)
# ---------------------------------------------------------------------------

def process_stationary(csv_bytes, *, resolution, algorithm, sampling_rate,
                       min_fixation_duration, detection_threshold, y_origin,
                       fixation_merge_threshold=None, adapt=False,
                       bg_image_bytes=None, bg_image_ext=None):
    """Run the screen-based detection pipeline on a CSV upload.

    Args:
        csv_bytes: Raw bytes of the uploaded CSV.
        resolution: ``(width, height)`` in pixels.
        algorithm: ``"idt"`` or ``"ivt"``.
        sampling_rate: Recording sampling rate in Hz.
        min_fixation_duration: Minimum fixation duration in ms.
        detection_threshold: px for I-DT, px/ms for I-VT.
        y_origin: One of ``top-left``, ``top-right``, ``bottom-left``,
            ``bottom-right``.
        fixation_merge_threshold: Merge fixations within this many pixels.
        adapt: Enable the adaptive threshold.
        bg_image_bytes: Optional background image bytes.
        bg_image_ext: Extension of that image ('png', '.jpg', ...).

    Raises:
        ValueError: On an unknown *y_origin*, or when detection produces no
            usable events.
    """
    if y_origin not in _VALID_Y_ORIGINS:
        raise ValueError(
            f"Invalid y_origin {y_origin!r}. Must be one of: {', '.join(_VALID_Y_ORIGINS)}"
        )

    gaze_data, column_mapping, is_normalized = load_csv_gaze_data(_as_bytes(csv_bytes))

    detector = EventDetection(
        gaze_data,
        resolution=resolution,
        column_mapping=column_mapping,
        is_normalized=is_normalized,
    )
    detector.process_event(
        min_fixation_duration=min_fixation_duration,
        detection_threshold=detection_threshold,
        algorithm=algorithm,
        sampling_rate=sampling_rate,
        fixation_merge_threshold=fixation_merge_threshold,
        adapt=adapt,
        gain=0.0,
        window_size_ms=0.0,
    )
    _require_events(detector)

    bg_image = (
        encode_image_bytes(_as_bytes(bg_image_bytes), bg_image_ext or "png")
        if bg_image_bytes else None
    )

    # Visualizations are produced from valid samples only
    valid_event_data = _valid_events(detector)
    visualizer = EyeTrackingVisualizer(valid_event_data, resolution=resolution)

    plot_html = visualizer.plot_gaze_points_and_fixations(
        bg_image_path=bg_image,
        aois=None,
        show_attach=False,
        y_origin=y_origin,
    )
    time_plot_html = visualizer.plot_gaze_with_time_scrolling(
        bg_image_path=bg_image,
        aois=None,
        time_window_ms=5000,
        step_ms=100,
        y_origin=y_origin,
    )

    return {
        "summary": _summarize_events(detector),
        "events_csv": detector.event_data_df.to_csv(index=False),
        "plot_html": plot_html,
        "time_plot_html": time_plot_html,
    }


def _summarize_events(detector):
    """Build the stat-card dict for a screen-based run."""
    df = detector.event_data_df
    return {
        "num_events": len(df),
        "num_fixations": int((df["event_type"] == "Fixation").sum()),
        "num_saccades": int((df["event_type"] == "Saccade").sum()),
        "num_fixation_points": int(df["fixation_id"].dropna().nunique()),
        "num_oor_gaze_points": int((df["event_type"] == "Out of Range Gaze Samples").sum()),
        "num_nan_gaze_points": int((df["event_type"] == "NaN").sum()),
        "best_threshold": getattr(detector, "best_threshold", None),
        "threshold_range": getattr(detector, "threshold_range", None),
    }


# ---------------------------------------------------------------------------
# Head-mounted (ZIP + video)
# ---------------------------------------------------------------------------

def process_head_mounted(zip_bytes, *, video_meta, video_url, resolution,
                         algorithm, sampling_rate, min_fixation_duration,
                         detection_threshold, adapt=False, gain=0.0,
                         window_size_ms=0.0):
    """Run detection + the video overlay for a head-mounted dataset.

    Args:
        zip_bytes: Raw bytes of the dataset ZIP (.npy for DD, .mat for GiW).
        video_meta: ``{fps, width, height, duration_s, n_frames}`` probed from
            the scene video by the host page — OpenCV cannot read video in
            Pyodide, so the container is parsed in JS instead.
        video_url: URL the overlay player should load the video from; a blob
            URL created by the host page from the user's local file.
        resolution: ``(width, height)`` fallback when *video_meta* lacks one.
        algorithm: ``"idt"`` or ``"ivt"``.
        sampling_rate: Gaze sampling rate in Hz.
        min_fixation_duration: Minimum fixation duration in ms.
        detection_threshold: px for I-DT, px/ms for I-VT.
        adapt: Enable the adaptive threshold.
        gain: Flow-RMS gain for the per-sample adaptive threshold.
        window_size_ms: Flow-RMS window in ms.

    Raises:
        ValueError: When the dataset cannot be loaded or detection fails.
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="opengazelab_"))
    zip_path = tmp_dir / "dataset.zip"
    # write_bytes accepts any buffer, so a large ZIP is never copied just to
    # satisfy a type check the way the CSV and image are (see _as_bytes).
    zip_path.write_bytes(zip_bytes)

    # The loaders call extract_video_metadata(video_path); register the
    # JS-probed values under the same key so no video file is ever needed.
    video_key = str(tmp_dir / "scene.mp4")
    meta = register_video_metadata(video_key, video_meta)

    fps = meta["fps"]
    vid_w = meta["width"] or resolution[0]
    vid_h = meta["height"] or resolution[1]

    try:
        gaze_df, metadata = load_head_mounted_dataset(
            str(zip_path),
            sampling_rate_hz=sampling_rate,
            video_path=video_key,
        )
    except Exception as exc:
        raise ValueError(f"Failed to load dataset: {exc}") from exc

    detector = EventDetection(
        gaze_df,
        resolution=(vid_w, vid_h),
        column_mapping=None,
        is_normalized=False,
    )
    detector.process_event(
        min_fixation_duration=min_fixation_duration,
        detection_threshold=detection_threshold,
        algorithm=algorithm,
        sampling_rate=sampling_rate,
        adapt=adapt,
        gain=gain,
        window_size_ms=window_size_ms,
        correct_timestamps_flag=False,
    )
    _require_events(detector)

    video_start_time = metadata.get("video_start_time", 0.0)
    flow_data = _build_flow_data(gaze_df, metadata, video_start_time)
    gt_labels = (
        gaze_df["gt_label"]
        if metadata.get("has_gt_labels") and "gt_label" in gaze_df.columns
        else None
    )

    valid_events = _valid_events(detector)
    video_plot_html = generate_video_gaze_visualization(
        event_df=valid_events,
        video_url=video_url,
        resolution=(vid_w, vid_h),
        fps=fps,
        video_start_time=video_start_time,
        gt_labels_series=gt_labels,
        flow_data=flow_data,
    )

    f1_fixation, f1_saccade = _compute_f1_scores(valid_events, gt_labels)

    return {
        "summary": _summarize_video_events(
            detector,
            fps=fps,
            resolution=(vid_w, vid_h),
            algorithm=algorithm,
            gt_labels=gt_labels,
            f1_fixation=f1_fixation,
            f1_saccade=f1_saccade,
        ),
        "events_csv": detector.event_data_df.to_csv(index=False),
        "video_plot_html": video_plot_html,
    }


def _summarize_video_events(detector, *, fps, resolution, algorithm, gt_labels,
                            f1_fixation, f1_saccade):
    """Build the stat-card dict for a head-mounted run."""
    df = detector.event_data_df
    vid_w, vid_h = resolution
    return {
        "num_events": len(df),
        "num_fixations": int((df["event_type"] == "Fixation").sum()),
        "num_saccades": int((df["event_type"] == "Saccade").sum()),
        "num_fixation_points": int(df["fixation_id"].dropna().nunique()),
        "fps": fps,
        "video_resolution": f"{vid_w}x{vid_h}",
        "algorithm": algorithm,
        "has_gt": gt_labels is not None,
        "best_threshold": getattr(detector, "best_threshold", None),
        "threshold_range": getattr(detector, "threshold_range", None),
        "f1_fixation": f1_fixation,
        "f1_saccade": f1_saccade,
    }


def _build_flow_data(gaze_df, metadata, video_start_time):
    """Downsample optical flow to ~one entry per video frame for the overlay JS.

    Times are emitted relative to ``video.currentTime`` (which starts at 0),
    not the raw epoch seconds in the .npy.
    """
    if "flow_x" not in gaze_df.columns or "flow_y" not in gaze_df.columns:
        return None

    n_video_frames = metadata.get("n_video_frames", len(gaze_df))
    step = max(1, len(gaze_df) // max(1, n_video_frames))
    flow_data = []
    for i in range(0, len(gaze_df), step):
        row = gaze_df.iloc[i]
        flow_data.append({
            "time_s": round(float(row["timestamp"]) - video_start_time, 4),
            "flow_x": round(float(row["flow_x"]), 3),
            "flow_y": round(float(row["flow_y"]), 3),
        })
    return flow_data


def _compute_f1_scores(valid_events, gt_labels):
    """Return ``(f1_fixation, f1_saccade)`` or ``(None, None)`` without GT."""
    if gt_labels is None:
        return None, None
    pred = (valid_events["event_type"] == "Fixation").astype(int).values
    gt_vals = gt_labels.loc[valid_events.index].values.astype(int)
    return (
        round(binary_f1(gt_vals, pred, pos_label=1), 4),
        round(binary_f1(gt_vals, pred, pos_label=0), 4),
    )


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

def _as_bytes(data):
    """Normalize any buffer to ``bytes``.

    Uploads arrive from JavaScript as ``Uint8Array``, which Pyodide converts to
    a ``memoryview``. Downstream code tests ``isinstance(..., bytes)`` to tell
    raw file content from a decoded string, and a memoryview fails that test —
    so the conversion happens once here at the boundary rather than being
    guarded for throughout the library.
    """
    return data if isinstance(data, bytes) else bytes(data)


def _require_events(detector):
    """Turn ``process_event``'s ``None`` result into a message the UI can show."""
    if getattr(detector, "event_data_df", None) is None:
        raise ValueError(
            "Detection produced no events. Check that the gaze coordinates fall "
            "within the given resolution and that the sampling rate is correct."
        )


def _valid_events(detector):
    """Drop NaN / out-of-range samples before visualizing."""
    df = detector.event_data_df
    return df[~df["event_type"].isin(["NaN", "Out of Range Gaze Samples"])].copy()
