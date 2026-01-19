"""
FastAPI backend for Eye Tracking Fixation Detection.
Handles CSV file uploads and processes them using the detection pipeline.
"""

import os
import json
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from datetime import datetime

# Import classes from src
from src import EventDetection, EyeTrackingVisualizer

app = FastAPI(
    title="Eye Tracking Fixation Detection API",
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
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
EVENTS_FOLDER.mkdir(parents=True, exist_ok=True)
VISUALIZATION_FOLDER.mkdir(parents=True, exist_ok=True)

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB


def detect_delimiter(file_path):
    """Detect the delimiter used in a CSV file."""
    delimiters = [';', ',', '\t', ' ']
    with open(file_path, 'r', encoding='utf-8') as f:
        first_line = f.readline()
    
    for delimiter in delimiters:
        if delimiter in first_line:
            return delimiter
    return ','  # Default to comma


@app.get("/api/status")
async def status():
    """Health check endpoint."""
    return {"status": "ok", "message": "Backend is running"}


@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    resolution: str = Form("2560,1440"),
    min_fixation_duration: int = Form(50),
    detect_threshold: float = Form(0.5),
    algorithm: str = Form("idt"),
    sampling_rate: int = Form(1000),
    fixation_merge_threshold: Optional[float] = Form(None),
    adapt: bool = Form(False)
):
    """
    Upload CSV file and parameters for processing.
    
    - **file**: CSV file with gaze data
    - **resolution**: Display resolution (e.g., "2560,1440")
    - **min_fixation_duration**: Minimum fixation duration in ms
    - **detect_threshold**: Detection threshold (0.0-1.0)
    - **algorithm**: Detection algorithm ('idt' or 'ivt')
    - **sampling_rate**: Sampling rate in Hz
    - **fixation_merge_threshold**: Maximum distance to merge fixations (pixels, optional)
    - **adapt**: Enable adaptive threshold adjustment (boolean)
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
        adapt=adapt
    )
    
    return {
        "success": True,
        "message": "File processed successfully",
        "filename": original_name,
        "result": result
    }


def process_gaze_data(file_path, resolution, min_fixation_duration, 
                     detect_threshold, algorithm, sampling_rate, output_name,
                     fixation_merge_threshold=None, adapt=False):
    """
    Process gaze data CSV file using EventDetection pipeline.
    
    Args:
        file_path: Path to uploaded CSV file
        resolution: Tuple of (width, height) display resolution
        min_fixation_duration: Minimum fixation duration in ms
        detect_threshold: Detection threshold
        algorithm: 'idt' or 'ivt'
        sampling_rate: Sampling rate in Hz
        output_name: Output filename prefix
        fixation_merge_threshold: Maximum distance to merge fixations (pixels, optional)
        adapt: Enable adaptive threshold adjustment (boolean)
    
    Returns:
        Dictionary with processing results
    """
    # Detect delimiter
    delimiter = detect_delimiter(file_path)
    
    # Read CSV
    gaze_data = pd.read_csv(file_path, sep=delimiter)
    
    # Initialize EventDetection with the gaze data
    detector = EventDetection(gaze_data, resolution=resolution)
    
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
    
    # Create visualization using EyeTrackingVisualizer class
    plot_file = VISUALIZATION_FOLDER / f"{output_name}_visualization.html"
    visualizer = EyeTrackingVisualizer(detector.event_data_df)
    visualizer.plot_gaze_points_and_fixations(
        str(plot_file),
        bg_image_path=None,
        aois=None,
        show_attach=False
    )
    
    return {
        'events_file': str(events_output_file.relative_to(BASE_DIR)),
        'plot_file': str(plot_file.relative_to(BASE_DIR)) if plot_file else None,
        'num_events': len(detector.event_data_df),
        'num_fixations': len(detector.event_data_df[detector.event_data_df['event_type'] == 'Fixation']),
        'num_saccades': len(detector.event_data_df[detector.event_data_df['event_type'] == 'Saccade'])
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


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000)
