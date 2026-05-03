# OpenGazeLab

A web-based application for processing eye-tracking gaze data to detect and classify **fixations**, **saccades**, and **blinks**. Supports both **stationary (screen-based) eye trackers** and **head-mounted eye trackers** (e.g. Pupil Invisible). Provides a Python processing pipeline and a modern web interface for researchers working with eye-tracking data.

> **Note**: Requires Python 3.11 or below.

---

## Features

### Two Operating Modes

| | Stationary Eye Tracker | Head-Mounted Eye Tracker |
|---|---|---|
| **Input** | CSV gaze file | ZIP of `.npy` files + MP4 video |
| **Algorithms** | I-DT, I-VT | I-VT with flow compensation (I-VAT+Frel) |
| **Flow compensation** | — | Optical flow removes head motion from gaze velocity |
| **Adaptive threshold** | MAD-based | Flow-RMS-based |
| **Visualization** | Stationary plot, time-scrolling plot | Video overlay with gaze + events |

### Data Processing
- **Automatic delimiter detection** (stationary mode): Supports `;`, `,`, `\t`, `|`, and space
- **Automatic column mapping** (stationary mode): Detects x/y/timestamp column names automatically
- **Coordinate normalization detection**: Auto-detects normalized (0-1) vs pixel coordinates
- **Timestamp correction**: Regularizes irregular timestamps to uniform intervals
- **Blink detection**: Identifies NaN/missing gaze data as blinks
- **Fixation merging**: Combines spatially close consecutive fixations using duration-weighted averaging
- **Optical flow integration** (head-mounted mode): Compensates for head/camera motion using per-frame optical flow grids

### Detection Algorithms
- **I-DT (Dispersion-Threshold)**: Classifies fixations based on spatial dispersion (stationary mode)
- **I-VT (Velocity-Threshold)**: Classifies fixations based on point-to-point velocity
- **I-VAT+Frel** (head-mounted mode): Enhanced I-VT that subtracts camera flow velocity from gaze velocity to isolate true eye movements. Uses Savitzky-Golay smoothing, relative velocity computation, and flow-RMS adaptive thresholding
- **Adaptive thresholding**: MAD-based (stationary) or flow-RMS-based (head-mounted)
- **Threshold optimization**: Uses Calinski-Harabasz score for automatic parameter tuning

### Visualization
- **Stationary visualization**: Interactive Plotly plot with gaze samples, fixations, and scanpath
- **Time-scrolling visualization**: Animated playback with play/pause controls and time slider
- **Video overlay visualization** (head-mounted mode): Gaze events overlaid on scene camera video with playback controls, optical flow vectors, and optional ground truth comparison
- **AOI overlay**: Display Areas of Interest on visualizations
- **Background image support**: Overlay stimulus screen image behind gaze plots (stationary mode)

### Web Interface
- **Mode toggle**: Switch between Stationary Eye Tracker and Head-Mounted Eye Tracker
- **Drag-and-drop upload**: Easy file selection for CSV, ZIP, video, and images
- **Parameter configuration**: Full control over detection parameters per mode
- **Real-time results**: Statistics display (fixations, saccades, blinks, F1 scores)
- **CSV export**: Download processed event data
- **Embedded visualizations**: View plots and video overlays directly in browser

---

## Quick Start

### Installation
```bash
cd event_detection-backend
pip install -r requirements.txt

cd event_detection-frontend
npm install
```

### Option A: Windows
Double-click `start_servers.bat` in the root directory. This opens two terminals and starts both servers automatically.

### Option B: Manual Start

**1. Start Backend:**
```bash
cd event_detection-backend
python main.py
```
Backend runs at: `http://127.0.0.1:5000`

**2. Start Frontend:**
```bash
cd event_detection-frontend
npm run start
```
Frontend opens at: `http://localhost:8000`

---

## Project Structure

```
event_detection/
├── start_servers.bat              # Windows startup script
├── README.md                      # This file
│
├── event_detection-backend/       # Python FastAPI backend
│   ├── main.py                    # API server (stationary + head-mounted endpoints)
│   ├── requirements.txt           # Python dependencies
│   ├── src/
│   │   ├── __init__.py            # EventDetection & EyeTrackingVisualizer classes
│   │   ├── detection_algorithms.py # I-DT, I-VT implementations
│   │   ├── pipeline.py            # Event detection pipeline
│   │   ├── preprocessing.py       # I-VAT+Frel pipeline (savgol, flow, adaptive)
│   │   ├── visualization.py       # Plotly visualizations
│   │   ├── dataset_loader.py      # .npy ZIP loader for head-mounted data
│   │   ├── utils.py               # Velocity, MAD, merging utilities
│   │   └── aoi.py                 # AOI classification
│   └── data/
│       ├── uploads/               # Uploaded CSV files
│       ├── images/                # Uploaded background images
│       ├── videos/                # Uploaded scene camera videos
│       ├── events/                # Processed event CSVs
│       └── visualization/         # HTML visualizations
│
└── event_detection-frontend/      # React frontend
    ├── index.html
    ├── package.json
    └── src/
        ├── App.js                 # Main React application (both modes)
        ├── App.css                # Component styles
        └── App.css              # Global styles
```

---

## Usage — Stationary Eye Tracker Mode

For screen-based / desktop eye trackers that produce CSV gaze data.

### Step 1: Upload CSV
Select the **Stationary Eye Tracker** tab. Click or drag-and-drop your gaze data CSV file into the upload area.

### Step 1b: Upload Background Image (Optional)
Optionally upload a screenshot or stimulus image (PNG, JPG, BMP, GIF, WebP) to display behind the gaze plots. The image is mapped to the full display resolution so gaze positions align with on-screen content.

### Step 2: Configure Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| **Algorithm** | I-DT | Detection algorithm: `I-DT` or `I-VT` |
| **Plot Origin** | Top-Left | Coordinate origin for visualization |
| **Display Resolution** | 2560,1440 | Screen resolution in pixels (width,height) |
| **Sampling Rate** | 250 Hz | Eye-tracker sampling rate |
| **Minimal Fixation Duration** | 50 ms | Minimum duration for valid fixation |
| **Detection Threshold** | 125 | For I-DT: dispersion in pixels. For I-VT: velocity in pixels/ms |
| **Merge Threshold** | None | Max distance (px) to merge nearby fixations |
| **Adaptive Threshold** | Off | Enable MAD-based adaptive threshold adjustment |

### Step 3: Process
Click **"Process Gaze Data"** and wait for processing.

### Step 4: View Results
- **Statistics**: Total events, fixation samples, saccade samples, blink points
- **Download CSV**: Get processed event data
- **Stationary Plot**: View gaze samples, fixations, and scanpath
- **Time-Scrolling Plot**: Watch fixations appear over time with playback controls

---

## Usage — Head-Mounted Eye Tracker Mode

For head-mounted eye trackers (e.g. Pupil Invisible) that produce `.npy` gaze data with optical flow and scene camera video.

### Step 1: Upload Dataset ZIP + Video
Select the **Head-Mounted Eye Tracker** tab. Upload two files:

1. **Dataset ZIP** — A `.zip` archive containing the required `.npy` files (see table below)
2. **Scene Camera Video** — An `.mp4` video recorded from the head-mounted camera

#### Required `.npy` Files in the ZIP

| File | Shape | Description |
|------|-------|-------------|
| `gaze.npy` | (N, 2) | Eye gaze position x, y in pixels |
| `time_gaze.npy` | (N,) | Gaze timestamps in seconds |
| `optic_flow.npy` | (M, 11, 11, 2) | Per-frame 11×11 optical flow grid |
| `time_optic_flow.npy` | (M,) | Optical flow frame timestamps in seconds |
| `time_scene_camera.npy` | (M,) | Scene camera frame timestamps in seconds |

#### Optional `.npy` Files

| File | Shape | Description |
|------|-------|-------------|
| `gt_labels.npy` | (N,) | Ground truth labels (1 = Fixation, 0 = Saccade). When present, F1 scores are computed automatically. |

The `.npy` files can be placed at the ZIP root or inside a single subfolder. Maximum ZIP size: 100 MB. Maximum video size: 500 MB.

### Step 2: Configure Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| **Video Resolution** | 1088,1080 | Scene camera resolution (width,height) |
| **Sampling Rate** | 30 Hz | Gaze sampling rate |
| **Min Fixation Duration** | 50 ms | Minimum duration for valid fixation |
| **Detection Threshold** | 1.0 | I-VT velocity threshold (px/ms) |
| **Adaptive Threshold** | On | Enable flow-RMS-based adaptive thresholding |

### Step 3: Process
Click **"Process Video Data"** and wait for processing.

### Step 4: View Results
- **Statistics**: fixation samples, saccade samples, duration, video resolution, FPS
- **F1 Scores**: Fixation and saccade F1 scores (when ground truth labels are provided)
- **Download CSV**: Get processed event data
- **Video Visualization**: Interactive video with gaze overlay, event markers, and optical flow vectors

---

## Detection Algorithms Explained

### I-DT (Dispersion-Threshold Identification)
Classifies gaze samples as fixations when **spatial dispersion** within a temporal window is below a threshold.

- **Dispersion formula**: `(max_x - min_x) + (max_y - min_y)` in pixels
- **Best for**: Low sampling rate data, noisy data, stationary eye trackers
- **Typical threshold**: 100–200 pixels

### I-VT (Velocity-Threshold Identification)
Classifies gaze samples as fixations when **point-to-point velocity** is below a threshold.

- **Velocity formula**: `sqrt(dx² + dy²) / dt` in pixels/ms
- **Best for**: High sampling rate data, clean data
- **Typical threshold**: 20–50 pixels/ms (stationary), ~1.0 px/ms (head-mounted)

### I-VAT+Frel (Head-Mounted Enhanced Pipeline)
An enhanced I-VT pipeline designed for head-mounted eye trackers where camera motion contaminates gaze velocity:

1. **Savitzky-Golay Smoothing** (55 ms window): Smooths raw gaze coordinates
2. **Gaze Velocity**: Computed on smoothed coordinates
3. **Flow Velocity**: Camera/head motion extracted from optical flow vectors
4. **Relative Velocity**: `vel_relative = vel_gaze − vel_flow` — isolates true eye movement
5. **Flow RMS**: Rolling-window RMS of flow velocity (quantifies head motion)
6. **Adaptive Threshold**: `threshold = base + gain × flow_rms` — raises threshold during head movement
7. **Classification**: Compare relative velocity against the (adaptive or fixed) threshold

### Adaptive Thresholding
- **Stationary mode** (MAD-based):
  ```
  adapted_threshold = original_threshold × (1 + tuning_parameter × MAD(velocity))
  ```
- **Head-mounted mode** (flow-RMS-based):
  ```
  threshold_i = base_threshold + gain × flow_rms_i
  ```
  Automatically raises the detection threshold during periods of head movement.

---

## Input Formats

### Stationary Mode — CSV

Your CSV file should contain columns for gaze coordinates and timestamps:

| Column | Required | Description |
|--------|----------|-------------|
| `x` or `gaze_x` | Yes | X coordinate (pixels or normalized 0-1) |
| `y` or `gaze_y` | Yes | Y coordinate (pixels or normalized 0-1) |
| `timestamp` or `time` | Yes | Time in milliseconds or seconds |

**Auto-detected delimiters**: `;`, `,`, `\t`, `|`, space

#### Example CSV
```csv
timestamp;x;y
1000;1280;720
1010;1281;720
1020;1282;720
1030;1500;800
1040;1501;801
```

### Head-Mounted Mode — NPY ZIP + MP4

See [Required `.npy` Files in the ZIP](#required-npy-files-in-the-zip) above. The dataset follows the Drews/Pupil Invisible format with gaze coordinates in pixel space, timestamps in seconds, and optical flow as 11×11 grids per video frame.

---

## Output Data Structure

Each gaze sample in the output CSV is classified with:

| Column | Description |
|--------|-------------|
| `x`, `y` | Original gaze coordinates (pixels) |
| `timestamp` | Time in milliseconds |
| `event_type` | `Fixation`, `Saccade`, or `Blink` |
| `fixation_x`, `fixation_y` | Fixation centroid coordinates |
| `fixation_id` | Unique fixation identifier |
| `saccade_id` | Unique saccade identifier |
| `blink_id` | Unique blink identifier |
| `event_duration` | Duration of the event in ms |
| `start_time`, `end_time` | Event temporal bounds |

---

## Troubleshooting

### Backend won't start
- Ensure Python 3.11 or below is installed
- Install dependencies: `pip install -r requirements.txt`
- Check if port 5000 is in use: try a different port
- Check console for error messages

### Frontend can't connect to backend
- Backend must be running on `http://127.0.0.1:5000`
- Frontend must be on `http://localhost:8000`
- Check browser console for CORS errors
- Restart both servers

### Processing fails
- Check CSV has required columns (x, y, timestamp)
- Verify data doesn't contain text in numeric columns
- Try different delimiter if auto-detection fails
- Check for NaN values in coordinate columns

### No fixations detected
- Try lowering the detection threshold
- Increase min_fixation_duration if data is noisy
- Switch between I-DT and I-VT algorithms
- Enable adaptive thresholding
