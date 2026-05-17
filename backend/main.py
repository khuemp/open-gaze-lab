"""FastAPI backend for OpenGazeLab.

Exposes two upload endpoints — one for screen-based CSVs, one for
head-mounted ZIP+MP4 datasets — plus result/asset retrieval routes.
The actual data preparation, detection, and visualization live in
``src/`` (see :mod:`src.preprocess_csv`, :mod:`src.preprocess_headmounted`,
:mod:`src.pipeline`, and :mod:`src.visualization`). The head-mounted
subpackage holds three loaders (DD, GiW) and a dispatcher that routes each
upload by inspecting the ZIP contents.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

from src import (
    EventDetection,
    EyeTrackingVisualizer,
    extract_video_metadata,
    generate_video_gaze_visualization,
    load_csv_gaze_data,
    load_head_mounted_dataset,
)


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="OpenGazeLab API",
    description="API for processing eye-tracking gaze data and detecting fixations",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Only outputs are persisted (events CSVs and visualization HTMLs/videos).
# Uploaded inputs (CSVs, background images, raw ZIPs) live in tempfiles for
# the lifetime of one request — videos are the exception, persisted so the
# overlay player can stream them.
BASE_DIR = Path(__file__).parent
EVENTS_FOLDER = BASE_DIR / "data" / "events"
VISUALIZATION_FOLDER = BASE_DIR / "data" / "visualization"
EVENTS_FOLDER.mkdir(parents=True, exist_ok=True)
VISUALIZATION_FOLDER.mkdir(parents=True, exist_ok=True)

MAX_FILE_SIZE = 50 * 1024 * 1024     # 50 MB CSV
MAX_IMAGE_SIZE = 20 * 1024 * 1024    # 20 MB background image
MAX_VIDEO_SIZE = 5 * 1024 * 1024 * 1024   # 5 GB scene video (GiW recordings can be ~4 GB)
MAX_ZIP_SIZE = 100 * 1024 * 1024     # 100 MB dataset ZIP


def _require_params(**params):
    """Raise HTTP 400 listing every missing required parameter."""
    missing = [name for name, value in params.items() if value is None or value == ""]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required parameters: {', '.join(missing)}",
        )


def _parse_resolution(resolution: str) -> tuple:
    try:
        width, height = map(int, resolution.split(","))
        return width, height
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid resolution format. Use 'width,height'")


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/api/status")
async def status():
    return {"status": "ok", "message": "Backend is running"}


# ---------------------------------------------------------------------------
# Screen-based (CSV) upload + processing
# ---------------------------------------------------------------------------

@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    resolution: Optional[str] = Form(None),
    min_fixation_duration: Optional[int] = Form(None),
    detection_threshold: Optional[float] = Form(None),
    algorithm: Optional[str] = Form(None),
    sampling_rate: Optional[int] = Form(None),
    y_origin: Optional[str] = Form(None),
    fixation_merge_threshold: Optional[float] = Form(None),
    adapt: bool = Form(False),
    background_image: Optional[UploadFile] = File(None),
):
    """Upload a CSV gaze file and run the screen-based detection pipeline."""
    _require_params(
        resolution=resolution,
        algorithm=algorithm,
        sampling_rate=sampling_rate,
        min_fixation_duration=min_fixation_duration,
        detection_threshold=detection_threshold,
        y_origin=y_origin,
    )

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV file")

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File size exceeds 50MB limit")

    if y_origin not in ("top-left", "top-right", "bottom-left", "bottom-right"):
        raise HTTPException(
            status_code=400,
            detail="Invalid y_origin. Must be one of: top-left, top-right, bottom-left, bottom-right",
        )

    width, height = _parse_resolution(resolution)
    output_name = Path(file.filename).stem

    # Optional background image — staged in a tempfile so the visualization
    # can read it (it gets embedded as base64), then removed.
    bg_tmp_path = None
    if background_image and background_image.filename:
        allowed_ext = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
        image_ext = Path(background_image.filename).suffix.lower()
        if image_ext not in allowed_ext:
            raise HTTPException(status_code=400, detail="Invalid image format. Use PNG, JPG, GIF, BMP, or WebP")
        image_content = await background_image.read()
        if len(image_content) > MAX_IMAGE_SIZE:
            raise HTTPException(status_code=400, detail="Image size exceeds 20MB limit")
        fd, bg_tmp_path = tempfile.mkstemp(suffix=image_ext, prefix="bg_")
        with os.fdopen(fd, "wb") as f:
            f.write(image_content)

    try:
        result = process_gaze_data(
            content,
            resolution=(width, height),
            min_fixation_duration=min_fixation_duration,
            detection_threshold=detection_threshold,
            algorithm=algorithm,
            sampling_rate=sampling_rate,
            output_name=output_name,
            fixation_merge_threshold=fixation_merge_threshold,
            adapt=adapt,
            bg_image_path=bg_tmp_path,
            y_origin=y_origin,
        )
    finally:
        if bg_tmp_path and os.path.exists(bg_tmp_path):
            os.remove(bg_tmp_path)

    return {
        "success": True,
        "message": "File processed successfully",
        "filename": output_name,
        "result": result,
    }


def process_gaze_data(csv_content, resolution, min_fixation_duration,
                      detection_threshold, algorithm, sampling_rate, output_name,
                      fixation_merge_threshold=None, bg_image_path=None,
                      *, adapt, y_origin):
    """Run detection + both visualizations for a screen-based CSV upload."""
    gaze_data, column_mapping, is_normalized = load_csv_gaze_data(csv_content)

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

    events_output_file = EVENTS_FOLDER / f"{output_name}_events.csv"
    detector.event_data_df.to_csv(events_output_file, index=False)

    # Visualizations are produced from valid samples only
    valid_event_data = detector.event_data_df[
        ~detector.event_data_df["event_type"].isin(["NaN", "Out of Range Gaze Samples"])
    ].copy()

    plot_file = VISUALIZATION_FOLDER / f"{output_name}_visualization.html"
    visualizer = EyeTrackingVisualizer(valid_event_data, resolution=resolution)
    visualizer.plot_gaze_points_and_fixations(
        str(plot_file),
        bg_image_path=bg_image_path,
        aois=None,
        show_attach=False,
        y_origin=y_origin,
    )

    time_plot_file = VISUALIZATION_FOLDER / f"{output_name}_time_visualization.html"
    visualizer.plot_gaze_with_time_scrolling(
        str(time_plot_file),
        bg_image_path=bg_image_path,
        aois=None,
        time_window_ms=5000,
        step_ms=100,
        y_origin=y_origin,
    )

    return _summarize_events(detector, events_output_file, plot_file, time_plot_file)


def _summarize_events(detector, events_output_file, plot_file, time_plot_file):
    """Build the JSON-friendly summary dict returned by /api/upload."""
    df = detector.event_data_df
    return {
        "events_file": str(events_output_file.relative_to(BASE_DIR)),
        "plot_file": str(plot_file.relative_to(BASE_DIR)) if plot_file else None,
        "time_plot_file": str(time_plot_file.relative_to(BASE_DIR)) if time_plot_file else None,
        "num_events": len(df),
        "num_fixations": int((df["event_type"] == "Fixation").sum()),
        "num_saccades": int((df["event_type"] == "Saccade").sum()),
        "num_fixation_points": int(df["fixation_id"].dropna().nunique()),
        "num_oor_gaze_points": int((df["event_type"] == "Out of Range Gaze Samples").sum()),
        "num_nan_gaze_points": int((df["event_type"] == "NaN").sum()),
        "best_threshold": getattr(detector, "best_threshold", None),
        "threshold_range": getattr(detector, "threshold_range", None),
    }


@app.get("/api/results/{filename}")
async def get_results_csv(filename: str):
    file_path = EVENTS_FOLDER / f"{filename}_events.csv"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type="text/csv", filename=f"{filename}_events.csv")


@app.get("/api/plot/{filename}")
async def get_plot(filename: str):
    plot_file = VISUALIZATION_FOLDER / f"{filename}_visualization.html"
    if not plot_file.exists():
        raise HTTPException(status_code=404, detail="Plot not found")
    return HTMLResponse(content=plot_file.read_text(encoding="utf-8"))


@app.get("/api/plot-time/{filename}")
async def get_time_plot(filename: str):
    plot_file = VISUALIZATION_FOLDER / f"{filename}_time_visualization.html"
    if not plot_file.exists():
        raise HTTPException(status_code=404, detail="Time plot not found")
    return HTMLResponse(content=plot_file.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Head-mounted (ZIP + MP4) upload + processing
# ---------------------------------------------------------------------------

@app.post("/api/upload-video")
async def upload_video_dataset(
    dataset_zip: UploadFile = File(...),
    video: UploadFile = File(...),
    resolution: Optional[str] = Form(None),
    min_fixation_duration: Optional[int] = Form(None),
    detection_threshold: Optional[float] = Form(None),
    algorithm: Optional[str] = Form(None),
    sampling_rate: Optional[int] = Form(None),
    adapt: bool = Form(False),
    gain: float = Form(0.0),
    window_size_ms: float = Form(0.0),
):
    """Upload a .zip of .npy files + an .mp4 video for head-mounted processing."""
    _require_params(
        resolution=resolution,
        sampling_rate=sampling_rate,
        min_fixation_duration=min_fixation_duration,
        detection_threshold=detection_threshold,
    )
    if not dataset_zip.filename or not dataset_zip.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Dataset must be a .zip file")
    if not video.filename or not video.filename.lower().endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Video must be an .mp4 file")

    zip_content = await dataset_zip.read()
    if len(zip_content) > MAX_ZIP_SIZE:
        raise HTTPException(status_code=400, detail="ZIP file exceeds 100MB limit")
    video_content = await video.read()
    if len(video_content) > MAX_VIDEO_SIZE:
        raise HTTPException(status_code=400, detail="Video file exceeds 5 GB limit")

    width, height = _parse_resolution(resolution)
    output_name = Path(video.filename).stem

    # Persist the video alongside its visualization so /api/video/<name> can stream it.
    video_save_name = f"{output_name}{Path(video.filename).suffix.lower()}"
    video_path = VISUALIZATION_FOLDER / video_save_name
    video_path.write_bytes(video_content)

    # Stage the ZIP in a tempdir; load_npy_dataset extracts what it needs.
    tmp_dir = Path(tempfile.mkdtemp(prefix="npy_"))
    zip_path = tmp_dir / dataset_zip.filename
    zip_path.write_bytes(zip_content)

    try:
        result = process_video_dataset(
            zip_path=str(zip_path),
            video_path=str(video_path),
            video_save_name=video_save_name,
            output_name=output_name,
            min_fixation_duration=min_fixation_duration,
            detection_threshold=detection_threshold,
            algorithm=algorithm,
            sampling_rate=sampling_rate,
            adapt=adapt,
            gain=gain,
            window_size_ms=window_size_ms,
            fallback_resolution=(width, height),
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return {
        "success": True,
        "message": "Video dataset processed successfully",
        "filename": output_name,
        "result": result,
    }


def process_video_dataset(zip_path, video_path, video_save_name, output_name,
                          min_fixation_duration, detection_threshold, algorithm,
                          sampling_rate, *, adapt, gain, window_size_ms,
                          fallback_resolution):
    """Run detection + video overlay visualization for a head-mounted dataset."""
    try:
        gaze_df, metadata = load_head_mounted_dataset(
            zip_path,
            sampling_rate_hz=sampling_rate,
            video_path=video_path,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load dataset: {e}")

    video_meta = extract_video_metadata(video_path)
    fps = video_meta.get("fps", 30.0)
    vid_w = video_meta.get("width", fallback_resolution[0])
    vid_h = video_meta.get("height", fallback_resolution[1])

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

    events_output_file = EVENTS_FOLDER / f"{output_name}_events.csv"
    detector.event_data_df.to_csv(events_output_file, index=False)

    video_start_time = metadata.get("video_start_time", 0.0)
    flow_data = _build_flow_data(gaze_df, metadata, video_start_time)
    gt_labels = (
        gaze_df["gt_label"]
        if metadata.get("has_gt_labels") and "gt_label" in gaze_df.columns
        else None
    )

    valid_events = detector.event_data_df[
        ~detector.event_data_df["event_type"].isin(["NaN", "Out of Range Gaze Samples"])
    ].copy()

    vis_path = VISUALIZATION_FOLDER / f"{output_name}_video_visualization.html"
    generate_video_gaze_visualization(
        event_df=valid_events,
        video_url=f"/api/video/{video_save_name}",
        resolution=(vid_w, vid_h),
        fps=fps,
        video_start_time=video_start_time,
        gt_labels_series=gt_labels,
        flow_data=flow_data,
        output_path=str(vis_path),
    )

    f1_fixation, f1_saccade = _compute_f1_scores(valid_events, gt_labels)

    return _summarize_video_events(
        detector,
        events_output_file,
        vis_path,
        fps=fps,
        resolution=(vid_w, vid_h),
        algorithm=algorithm,
        gt_labels=gt_labels,
        f1_fixation=f1_fixation,
        f1_saccade=f1_saccade,
        video_filename=video_save_name,
    )


def _summarize_video_events(detector, events_output_file, vis_path, *,
                            fps, resolution, algorithm, gt_labels,
                            f1_fixation, f1_saccade, video_filename):
    """Build the JSON-friendly summary dict returned by /api/upload-video."""
    df = detector.event_data_df
    vid_w, vid_h = resolution
    return {
        "events_file": str(events_output_file.relative_to(BASE_DIR)),
        "video_plot_file": str(vis_path.relative_to(BASE_DIR)),
        "num_events": len(df),
        "num_fixations": int((df["event_type"] == "Fixation").sum()),
        "num_saccades": int((df["event_type"] == "Saccade").sum()),
        "num_fixation_points": int(df["fixation_id"].dropna().nunique()),
        "num_fixation_centers": int(df["fixation_id"].dropna().nunique()),
        "fps": fps,
        "video_resolution": f"{vid_w}x{vid_h}",
        "algorithm": algorithm,
        "has_gt": gt_labels is not None,
        "best_threshold": getattr(detector, "best_threshold", None),
        "threshold_range": getattr(detector, "threshold_range", None),
        "f1_fixation": f1_fixation,
        "f1_saccade": f1_saccade,
        "video_filename": video_filename,
    }


def _build_flow_data(gaze_df, metadata, video_start_time):
    """Downsample optical flow to ~one entry per video frame for the overlay JS.

    Times are emitted relative to ``video.currentTime`` (which starts at 0),
    not the raw epoch seconds in the .npy.
    """
    if "flow_x" not in gaze_df.columns or "flow_y" not in gaze_df.columns:
        return None

    n_video_frames = metadata.get("n_video_frames", len(gaze_df))
    step = max(1, len(gaze_df) // n_video_frames)
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
    from sklearn.metrics import f1_score
    pred = (valid_events["event_type"] == "Fixation").astype(int).values
    gt_vals = gt_labels.loc[valid_events.index].values.astype(int)
    return (
        round(float(f1_score(gt_vals, pred, pos_label=1, zero_division=0)), 4),
        round(float(f1_score(gt_vals, pred, pos_label=0, zero_division=0)), 4),
    )


@app.get("/api/video/{filename}")
async def serve_video(filename: str, request: Request):
    """Serve a video with HTTP Range support so the player can seek."""
    safe_name = Path(filename).name  # block path traversal
    video_path = VISUALIZATION_FOLDER / safe_name
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")

    file_size = video_path.stat().st_size
    range_header = request.headers.get("range")
    if not range_header:
        return FileResponse(video_path, media_type="video/mp4")

    # Parse "bytes=start-end"
    try:
        range_spec = range_header.replace("bytes=", "")
        parts = range_spec.split("-")
        start = int(parts[0])
        end = int(parts[1]) if parts[1] else file_size - 1
    except (ValueError, IndexError):
        raise HTTPException(status_code=416, detail="Invalid range header")

    end = min(end, file_size - 1)
    length = end - start + 1

    def iter_chunk():
        with open(video_path, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(65536, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(length),
        "Content-Type": "video/mp4",
    }
    return StreamingResponse(iter_chunk(), status_code=206, headers=headers)


@app.get("/api/plot-video/{filename}")
async def get_video_plot(filename: str):
    plot_file = VISUALIZATION_FOLDER / f"{filename}_video_visualization.html"
    if not plot_file.exists():
        raise HTTPException(status_code=404, detail="Video plot not found")
    return HTMLResponse(content=plot_file.read_text(encoding="utf-8"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000)
