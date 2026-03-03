"""
FastAPI backend for Eye-tracking Event Detection.
Handles CSV file uploads and processes them using the detection pipeline.
"""

import os
import json
import shutil
import tempfile
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from datetime import datetime

# Import classes from src
from src import EventDetection, EyeTrackingVisualizer
from src import load_npy_dataset, extract_video_metadata, generate_video_gaze_visualization

app = FastAPI(
    title="Eye-tracking Event Detection API",
    description="API for processing eye-tracking gaze data and detecting fixations",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure paths
BASE_DIR = Path(__file__).parent
UPLOAD_FOLDER = BASE_DIR / 'data' / 'uploads'
EVENTS_FOLDER = BASE_DIR / 'data' / 'events'
VISUALIZATION_FOLDER = BASE_DIR / 'data' / 'visualization'
IMAGES_FOLDER = BASE_DIR / 'data' / 'images'
VIDEOS_FOLDER = BASE_DIR / 'data' / 'videos'
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
EVENTS_FOLDER.mkdir(parents=True, exist_ok=True)
VISUALIZATION_FOLDER.mkdir(parents=True, exist_ok=True)
IMAGES_FOLDER.mkdir(parents=True, exist_ok=True)
VIDEOS_FOLDER.mkdir(parents=True, exist_ok=True)

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MAX_IMAGE_SIZE = 20 * 1024 * 1024  # 20MB for images
MAX_VIDEO_SIZE = 500 * 1024 * 1024  # 500MB for video files
MAX_ZIP_SIZE = 100 * 1024 * 1024  # 100MB for dataset ZIP


def detect_delimiter(file_path):
    """Detect the delimiter used in a CSV file."""
    delimiters = ['|', ';', ',', '\t', ' ']
    with open(file_path, 'r', encoding='utf-8') as f:
        first_line = f.readline()
    
    for delimiter in delimiters:
        if delimiter in first_line:
            return delimiter
    return ','  # Default to comma


def detect_column_mapping(df):
    """Detect which columns contain x, y coordinates and timestamp.
    
    Returns a dict with keys 'x', 'y', 'timestamp' mapping to actual column names.
    Also detects optional optical-flow columns ('flow_x', 'flow_y') and
    'video_timestamp' for head-motion compensation.
    """
    mapping = {}
    columns_lower = {col.lower(): col for col in df.columns}
    
    # Detect timestamp column
    timestamp_candidates = ['timestamp', 'time', 't']
    for candidate in timestamp_candidates:
        if candidate in columns_lower:
            mapping['timestamp'] = columns_lower[candidate]
            break
    
    # Detect x coordinate column
    x_candidates = ['x', 'gaze_x', 'gazex', 'pos_x', 'posx']
    for candidate in x_candidates:
        if candidate in columns_lower:
            mapping['x'] = columns_lower[candidate]
            break
    
    # Detect y coordinate column
    y_candidates = ['y', 'gaze_y', 'gazey', 'pos_y', 'posy']
    for candidate in y_candidates:
        if candidate in columns_lower:
            mapping['y'] = columns_lower[candidate]
            break
    
    # Detect optical-flow columns (for I-VAT+Frel head-motion compensation)
    flow_x_candidates = ['flow_x', 'optical_flow_x', 'of_x', 'optic_flow_x']
    for candidate in flow_x_candidates:
        if candidate in columns_lower:
            mapping['flow_x'] = columns_lower[candidate]
            break
    
    flow_y_candidates = ['flow_y', 'optical_flow_y', 'of_y', 'optic_flow_y']
    for candidate in flow_y_candidates:
        if candidate in columns_lower:
            mapping['flow_y'] = columns_lower[candidate]
            break
    
    # Detect video_timestamp column (used for flow velocity calculation)
    video_ts_candidates = ['video_timestamp', 'video_time', 'frame_timestamp']
    for candidate in video_ts_candidates:
        if candidate in columns_lower:
            mapping['video_timestamp'] = columns_lower[candidate]
            break
    
    return mapping


def detect_if_normalized(df, x_col, y_col):
    """Detect if coordinates are normalized (0-1) or in pixels.
    
    Returns True if normalized, False if in pixels.
    """
    x_values = df[x_col].dropna()
    y_values = df[y_col].dropna()
    
    if len(x_values) == 0 or len(y_values) == 0:
        return True  # Default to normalized
    
    x_max = x_values.max()
    y_max = y_values.max()
    
    # If max values > 2, assume pixel coordinates
    return not (x_max > 2 or y_max > 2)


@app.get("/api/status")
async def status():
    """Health check endpoint."""
    return {"status": "ok", "message": "Backend is running"}


@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    resolution: str = Form("2560,1440"),
    min_fixation_duration: int = Form(50),
    detect_threshold: float = Form(125),
    algorithm: str = Form("idt"),
    sampling_rate: int = Form(250),
    fixation_merge_threshold: Optional[float] = Form(None),
    adapt: bool = Form(False),
    background_image: Optional[UploadFile] = File(None),
    y_origin: str = Form("top-left")
):
    """
    Upload CSV file and parameters for processing.
    
    - **file**: CSV file with gaze data
    - **resolution**: Display resolution (e.g., "2560,1440")
    - **min_fixation_duration**: Minimum fixation duration in ms
    - **detect_threshold**: Detection threshold in pixels (dispersion for I-DT, velocity for I-VT)
    - **algorithm**: Detection algorithm ('idt' or 'ivt')
    - **sampling_rate**: Sampling rate in Hz
    - **fixation_merge_threshold**: Maximum distance to merge fixations (pixels, optional)
    - **adapt**: Enable adaptive threshold adjustment (boolean)
    - **background_image**: Optional background image file for visualization
    - **y_origin**: Origin position for plot axes ('top-left', 'top-right', 'bottom-left', 'bottom-right')
    """
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV file")
    
    # Check file size
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File size exceeds 50MB limit")
    
    # Validate y_origin
    valid_origins = ('top-left', 'top-right', 'bottom-left', 'bottom-right')
    if y_origin not in valid_origins:
        raise HTTPException(status_code=400, detail=f"Invalid y_origin. Must be one of: {', '.join(valid_origins)}")

    # Parse resolution
    try:
        width, height = map(int, resolution.split(','))
    except:
        raise HTTPException(status_code=400, detail="Invalid resolution format. Use 'width,height'")
    
    # Save uploaded file
    filename = f"{file.filename}"
    file_path = UPLOAD_FOLDER / filename
    
    # Get original filename without extension for output naming
    original_name = Path(file.filename).stem
    
    with open(file_path, 'wb') as f:
        f.write(content)
    
    # Handle background image if provided
    bg_image_path = None
    if background_image and background_image.filename:
        # Validate image type
        allowed_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
        image_ext = Path(background_image.filename).suffix.lower()
        if image_ext not in allowed_extensions:
            raise HTTPException(status_code=400, detail="Invalid image format. Use PNG, JPG, GIF, BMP, or WebP")
        
        # Read and validate image size
        image_content = await background_image.read()
        if len(image_content) > MAX_IMAGE_SIZE:
            raise HTTPException(status_code=400, detail="Image size exceeds 20MB limit")
        
        # Save the image
        image_filename = f"{original_name}_bg{image_ext}"
        bg_image_path = IMAGES_FOLDER / image_filename
        with open(bg_image_path, 'wb') as f:
            f.write(image_content)
    
    # Process the file
    result = process_gaze_data(
        file_path,
        resolution=(width, height),
        min_fixation_duration=min_fixation_duration,
        detect_threshold=detect_threshold,
        algorithm=algorithm,
        sampling_rate=sampling_rate,
        output_name=original_name,
        fixation_merge_threshold=fixation_merge_threshold,
        adapt=adapt,
        bg_image_path=str(bg_image_path) if bg_image_path else None,
        y_origin=y_origin
    )
    
    return {
        "success": True,
        "message": "File processed successfully",
        "filename": original_name,
        "result": result
    }


def process_gaze_data(file_path, resolution, min_fixation_duration, 
                     detect_threshold, algorithm, sampling_rate, output_name,
                     fixation_merge_threshold=None, adapt=False, bg_image_path=None,
                     y_origin='top-left'):
    """
    Process gaze data CSV file using EventDetection pipeline.
    
    Args:
        file_path: Path to uploaded CSV file
        resolution: Tuple of (width, height) display resolution
        min_fixation_duration: Minimum fixation duration in ms
        detect_threshold: Detection threshold in pixels
        algorithm: 'idt' or 'ivt'
        sampling_rate: Sampling rate in Hz
        output_name: Output filename prefix
        fixation_merge_threshold: Maximum distance to merge fixations (pixels, optional)
        adapt: Enable adaptive threshold adjustment (boolean)
        bg_image_path: Path to background image for visualization (optional)
        y_origin: Origin position for plot axes ('top-left', 'top-right', 'bottom-left', 'bottom-right')
    
    Returns:
        Dictionary with processing results
    """
    # Detect delimiter
    delimiter = detect_delimiter(file_path)
    
    # Read CSV
    gaze_data = pd.read_csv(file_path, sep=delimiter)
    
    # Detect column mapping and normalization
    column_mapping = detect_column_mapping(gaze_data)
    
    # Validate required columns exist
    if 'x' not in column_mapping or 'y' not in column_mapping:
        raise ValueError(f"Could not detect x/y coordinate columns. Found columns: {list(gaze_data.columns)}")
    if 'timestamp' not in column_mapping:
        raise ValueError(f"Could not detect timestamp column. Found columns: {list(gaze_data.columns)}")
    
    is_normalized = detect_if_normalized(gaze_data, column_mapping['x'], column_mapping['y'])
    
    # Initialize EventDetection with the gaze data and column mapping
    detector = EventDetection(
        gaze_data, 
        resolution=resolution, 
        column_mapping=column_mapping,
        is_normalized=is_normalized
    )
    
    # Process events using the class method
    detector.process_event(
        output_dir=str(EVENTS_FOLDER),
        min_fixation_duration=min_fixation_duration,
        detect_threshold=detect_threshold,
        algorithm=algorithm,
        sampling_rate=sampling_rate,
        fixation_merge_threshold=fixation_merge_threshold,
        adapt=adapt
    )
    
    # Save events CSV
    events_output_file = EVENTS_FOLDER / f"{output_name}_events.csv"
    detector.event_data_df.to_csv(events_output_file, index=False)
    
    # Create standard visualization using EyeTrackingVisualizer class (valid data only)
    valid_event_data = detector.event_data_df[
        ~detector.event_data_df['event_type'].isin(['NaN', 'Out of Range Gaze Points'])
    ].copy()
    
    plot_file = VISUALIZATION_FOLDER / f"{output_name}_visualization.html"
    visualizer = EyeTrackingVisualizer(valid_event_data, resolution=resolution)
    visualizer.plot_gaze_points_and_fixations(
        str(plot_file),
        bg_image_path=bg_image_path,
        aois=None,
        show_attach=False,
        y_origin=y_origin
    )
    
    # Create time-scrolling visualization
    time_plot_file = VISUALIZATION_FOLDER / f"{output_name}_time_visualization.html"
    visualizer.plot_gaze_with_time_scrolling(
        str(time_plot_file),
        bg_image_path=bg_image_path,
        aois=None,
        time_window_ms=5000,
        step_ms=100,
        y_origin=y_origin
    )
    
    return {
        'events_file': str(events_output_file.relative_to(BASE_DIR)),
        'plot_file': str(plot_file.relative_to(BASE_DIR)) if plot_file else None,
        'time_plot_file': str(time_plot_file.relative_to(BASE_DIR)) if time_plot_file else None,
        'num_events': len(detector.event_data_df),
        'num_fixations': len(detector.event_data_df[detector.event_data_df['event_type'] == 'Fixation']),
        'num_saccades': len(detector.event_data_df[detector.event_data_df['event_type'] == 'Saccade']),
        'num_fixation_points': detector.event_data_df['fixation_id'].dropna().nunique(),
        'num_oor_gaze_points': len(detector.event_data_df[detector.event_data_df['event_type'] == 'Out of Range Gaze Points']),
        'num_nan_gaze_points': len(detector.event_data_df[detector.event_data_df['event_type'] == 'NaN']),
        'best_threshold': detector.best_threshold if hasattr(detector, 'best_threshold') else None
    }


@app.get("/api/results/{filename}")
async def get_results_csv(filename: str):
    """Download events CSV file."""
    file_path = EVENTS_FOLDER / f"{filename}_events.csv"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        file_path,
        media_type='text/csv',
        filename=f"{filename}_events.csv"
    )


@app.get("/api/plot/{filename}")
async def get_plot(filename: str):
    """Get the HTML plot content."""
    plot_file = VISUALIZATION_FOLDER / f"{filename}_visualization.html"
    if not plot_file.exists():
        raise HTTPException(status_code=404, detail="Plot not found")
    
    with open(plot_file, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    return HTMLResponse(content=html_content)


@app.get("/api/plot-time/{filename}")
async def get_time_plot(filename: str):
    """Get the time-scrolling HTML plot content."""
    plot_file = VISUALIZATION_FOLDER / f"{filename}_time_visualization.html"
    if not plot_file.exists():
        raise HTTPException(status_code=404, detail="Time plot not found")
    
    with open(plot_file, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    return HTMLResponse(content=html_content)


# ---------------------------------------------------------------------------
# HEAD-MOUNTED EYE-TRACKER  endpoints
# ---------------------------------------------------------------------------


@app.post("/api/upload-video")
async def upload_video_dataset(
    dataset_zip: UploadFile = File(...),
    video: UploadFile = File(...),
    resolution: str = Form("1088,1080"),
    min_fixation_duration: int = Form(50),
    detect_threshold: float = Form(1.0),
    sampling_rate: int = Form(248),
    adapt: bool = Form(True),
):
    """
    Upload a .zip of .npy files + an .mp4 video for head-mounted processing.

    - **dataset_zip**: ZIP containing .npy files (gaze.npy, optic_flow.npy, etc.)
    - **video**: MP4 video from the scene camera
    - **resolution**: Video resolution (e.g. "1088,1080")
    - **min_fixation_duration**: Minimum fixation duration ms
    - **detect_threshold**: I-VT velocity threshold
    - **sampling_rate**: Gaze sampling rate Hz
    - **adapt**: Use adaptive threshold (boolean)
    """
    # --- validate files ---
    if not dataset_zip.filename or not dataset_zip.filename.lower().endswith('.zip'):
        raise HTTPException(status_code=400, detail="Dataset must be a .zip file")
    if not video.filename or not video.filename.lower().endswith('.mp4'):
        raise HTTPException(status_code=400, detail="Video must be an .mp4 file")

    zip_content = await dataset_zip.read()
    if len(zip_content) > MAX_ZIP_SIZE:
        raise HTTPException(status_code=400, detail="ZIP file exceeds 100MB limit")
    video_content = await video.read()
    if len(video_content) > MAX_VIDEO_SIZE:
        raise HTTPException(status_code=400, detail="Video file exceeds 500MB limit")

    try:
        width, height = map(int, resolution.split(','))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid resolution format. Use 'width,height'")

    # Derive an output name from the video filename
    output_name = Path(video.filename).stem

    # Save video
    video_path = VIDEOS_FOLDER / video.filename
    with open(video_path, 'wb') as f:
        f.write(video_content)

    # Save & extract ZIP into a temp directory
    tmp_dir = Path(tempfile.mkdtemp(prefix="npy_"))
    zip_path = tmp_dir / dataset_zip.filename
    with open(zip_path, 'wb') as f:
        f.write(zip_content)

    try:
        gaze_df, metadata = load_npy_dataset(str(zip_path))
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"Failed to load dataset: {e}")

    # Get video metadata (fps, actual resolution)
    video_meta = extract_video_metadata(str(video_path))
    fps = video_meta.get("fps", 30.0)
    vid_w = video_meta.get("width", width)
    vid_h = video_meta.get("height", height)

    # Run EventDetection pipeline
    detector = EventDetection(
        gaze_df,
        resolution=(vid_w, vid_h),
        column_mapping=None,
        is_normalized=False,
    )
    detector.process_event(
        output_dir=str(EVENTS_FOLDER),
        min_fixation_duration=min_fixation_duration,
        detect_threshold=detect_threshold,
        algorithm="ivt",
        sampling_rate=sampling_rate,
        adapt=adapt,
    )

    events_output_file = EVENTS_FOLDER / f"{output_name}_events.csv"
    detector.event_data_df.to_csv(events_output_file, index=False)

    # Build flow data list for visualization
    flow_data = None
    if "flow_x" in gaze_df.columns and "flow_y" in gaze_df.columns:
        # Downsample flow to ~one per video frame
        step = max(1, len(gaze_df) // metadata.get("n_video_frames", len(gaze_df)))
        flow_data = []
        for i in range(0, len(gaze_df), step):
            row = gaze_df.iloc[i]
            flow_data.append({
                "time_s": round(float(row["timestamp"]), 4),
                "flow_x": round(float(row["flow_x"]), 3),
                "flow_y": round(float(row["flow_y"]), 3),
            })

    # Ground truth labels (if present)
    gt_labels = None
    if metadata.get("has_gt_labels") and "gt_label" in gaze_df.columns:
        gt_labels = gaze_df["gt_label"]

    # Valid events only
    valid_events = detector.event_data_df[
        ~detector.event_data_df['event_type'].isin(['NaN', 'Out of Range Gaze Points'])
    ].copy()

    video_url = f"/api/video/{video.filename}"
    video_start_time = metadata.get("video_start_time", 0.0)

    vis_path = VISUALIZATION_FOLDER / f"{output_name}_video_visualization.html"
    generate_video_gaze_visualization(
        event_df=valid_events,
        video_url=video_url,
        resolution=(vid_w, vid_h),
        fps=fps,
        video_start_time=video_start_time,
        gt_labels_series=gt_labels,
        flow_data=flow_data,
        output_path=str(vis_path),
    )

    shutil.rmtree(tmp_dir, ignore_errors=True)

    # Compute duration from valid events
    timestamps_s = valid_events["timestamp"].values / 1000.0
    duration_s = round(float(timestamps_s[-1] - timestamps_s[0]), 2) if len(timestamps_s) > 1 else 0.0

    # Compute F1 scores if ground truth is available
    f1_fixation = None
    f1_saccade = None
    if gt_labels is not None:
        from sklearn.metrics import f1_score
        pred = (valid_events["event_type"] == "Fixation").astype(int).values
        gt_vals = gt_labels.loc[valid_events.index].values.astype(int)
        f1_fixation = round(float(f1_score(gt_vals, pred, pos_label=1, zero_division=0)), 4)
        f1_saccade = round(float(f1_score(gt_vals, pred, pos_label=0, zero_division=0)), 4)

    return {
        "success": True,
        "message": "Video dataset processed successfully",
        "filename": output_name,
        "result": {
            "events_file": str(events_output_file.relative_to(BASE_DIR)),
            "video_plot_file": str(vis_path.relative_to(BASE_DIR)),
            "num_events": len(detector.event_data_df),
            "num_fixations": len(detector.event_data_df[detector.event_data_df['event_type'] == 'Fixation']),
            "num_saccades": len(detector.event_data_df[detector.event_data_df['event_type'] == 'Saccade']),
            "num_fixation_points": int(detector.event_data_df['fixation_id'].dropna().nunique()),
            "num_fixation_centers": int(detector.event_data_df['fixation_id'].dropna().nunique()),
            "fps": fps,
            "video_resolution": f"{vid_w}x{vid_h}",
            "has_gt": gt_labels is not None,
            "best_threshold": detector.best_threshold if hasattr(detector, 'best_threshold') else None,
            "duration_s": duration_s,
            "f1_fixation": f1_fixation,
            "f1_saccade": f1_saccade,
            "video_filename": video.filename,
        },
    }


@app.get("/api/video/{filename}")
async def serve_video(filename: str, request: Request):
    """Serve a video file with HTTP Range support for seeking."""
    # Sanitize filename to prevent path traversal
    safe_name = Path(filename).name
    video_path = VIDEOS_FOLDER / safe_name
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")

    file_size = video_path.stat().st_size
    range_header = request.headers.get("range")

    if range_header:
        # Parse Range header: "bytes=start-end"
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

    # No range requested – stream entire file
    return FileResponse(video_path, media_type="video/mp4")


@app.get("/api/plot-video/{filename}")
async def get_video_plot(filename: str):
    """Get the head-mounted video visualization HTML."""
    plot_file = VISUALIZATION_FOLDER / f"{filename}_video_visualization.html"
    if not plot_file.exists():
        raise HTTPException(status_code=404, detail="Video plot not found")

    with open(plot_file, 'r', encoding='utf-8') as f:
        html_content = f.read()

    return HTMLResponse(content=html_content)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000)
