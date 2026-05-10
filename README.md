# OpenGazeLab

A web-based toolkit for processing eye-tracking gaze data and classifying it into **fixations** (eyes holding still on a target) and **saccades** (rapid eye movements between targets). Supports both **stationary eye trackers** and **head-mounted eye trackers**. Provides a Python processing pipeline and a browser-based UI for researchers working with eye-tracking data.

> **Note**: Python 3.10 recommended.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [How to Use — Stationary Eye Tracker](#how-to-use--stationary-eye-tracker)
3. [How to Use — Head-Mounted Eye Tracker](#how-to-use--head-mounted-eye-tracker)
4. [Recommended Parameters](#recommended-parameters)
5. [Output Reference](#output-reference)
6. [Toolkit Structure](#toolkit-structure)
7. [How the Pipeline Works](#how-the-pipeline-works)
8. [Detection Algorithms Explained](#detection-algorithms-explained)
9. [Features Overview](#features-overview)
10. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Installation
```bash
cd backend
pip install -r requirements.txt

cd frontend
npm install
```

### Run the App

**Option A — Windows:** Double-click [start_servers.bat](start_servers.bat) in the project root. Two terminals open and start both servers automatically.

**Option B — Manual:**
```bash
# Terminal 1 — Backend
cd backend
python main.py
# → Backend runs at http://127.0.0.1:5000

# Terminal 2 — Frontend
cd frontend
npm run start
# → Frontend opens at http://localhost:8000
```

Open the frontend URL in your browser, pick a mode (Stationary or Head-Mounted), and follow the steps below.

---

## How to Use — Stationary Eye Tracker

For screen-based / desktop eye trackers that record gaze coordinates in a CSV file.

### Input

| What | Required | Description |
|------|----------|-------------|
| **Gaze CSV file** | Yes | One sample per row with x, y, and timestamp columns |
| **Background image** | No | Stimulus screenshot (PNG/JPG/BMP/GIF/WebP) — overlaid behind the gaze plot for context |

#### CSV format

The CSV needs three columns. Names are detected automatically (common variants supported):

| Column meaning | Accepted names | Units |
|----------------|----------------|-------|
| Horizontal gaze | `x`, `gaze_x`, `X`, … | pixels OR normalized (0–1) — auto-detected |
| Vertical gaze | `y`, `gaze_y`, `Y`, … | pixels OR normalized (0–1) — auto-detected |
| Time | `timestamp`, `time`, … | milliseconds, seconds, or epoch — auto-detected |

**Auto-detected delimiters**: `;`, `,`, `\t`, `|`, space.

Example:
```csv
timestamp;x;y
1000;1280;720
1010;1281;720
1020;1282;720
1030;1500;800
1040;1501;801
```

### Configuration

| Parameter | Description |
|-----------|-------------|
| **Algorithm** | `I-DT` (dispersion-based) or `I-VT` (velocity-based) |
| **Y-Origin** | Coordinate origin convention for visualization |
| **Display Resolution** | Screen resolution in pixels (width,height) |
| **Sampling Rate** | Eye-tracker sampling rate |
| **Min Fixation Duration** | Minimum duration to count as a fixation |
| **Detection Threshold** | I-DT: dispersion in pixels. I-VT: velocity in px/ms |
| **Merge Threshold** | Max distance (px) to merge nearby fixations |
| **Adaptive Threshold** | Enable MAD-based adaptive thresholding |

### Output

After clicking **"Process Gaze Data"**, you get:

- **Statistics panel** — total events, fixation samples, saccade samples, invalid samples
- **Downloadable CSV** — original data plus event classification columns ([see Output Reference](#output-reference))
- **Stationary plot** (interactive Plotly HTML) — gaze samples colored by event type, fixation centers numbered in scan order, scanpath lines, optional background image
- **Time-scrolling plot** (animated Plotly HTML) — playback of fixations/saccades over time with play/pause controls and a time slider

---

## How to Use — Head-Mounted Eye Tracker

For head-mounted eye trackers that record gaze data, optical flow from the scene camera, and a scene video.

### Input

Upload **two files**:

1. **Dataset ZIP** (max 100 MB) — archive of `.npy` arrays
2. **Scene Camera Video** (max 500 MB) — `.mp4` from the head-mounted camera

#### Required files inside the ZIP

| File | Shape | Description |
|------|-------|-------------|
| `gaze.npy` | (N, 2) | Eye gaze position (x, y) in pixels |
| `time_gaze.npy` | (N,) | Gaze timestamps in seconds |
| `optic_flow.npy` | (M, 11, 11, 2) | Per-frame 11×11 optical flow grid |
| `time_optic_flow.npy` | (M,) | Optical flow frame timestamps in seconds |
| `time_scene_camera.npy` | (M,) | Scene camera frame timestamps in seconds |

#### Optional files

| File | Shape | Description |
|------|-------|-------------|
| `gt_labels.npy` | (N,) | Ground truth labels (1 = Fixation, 0 = Saccade). Triggers automatic F1 score computation. |

`.npy` files may live at the ZIP root or inside a single subfolder.

### Configuration

| Parameter | Description |
|-----------|-------------|
| **Algorithm** | `I-DT` (relative dispersion) or `I-VT` (relative velocity) — both run with optical-flow compensation |
| **Video Resolution** | Scene camera resolution (width,height) |
| **Sampling Rate** | Gaze sampling rate |
| **Min Fixation Duration** | Minimum duration to count as a fixation |
| **Detection Threshold** | I-VT: relative-velocity threshold (px/ms). I-DT: relative-dispersion threshold (px). See [Recommended Parameters](#recommended-parameters) |
| **Adaptive Threshold** | Enable flow-RMS-based adaptive thresholding |

### Output

After clicking **"Process Video Data"**, you get:

- **Statistics panel** — fixation samples, saccade samples, total duration, video resolution, FPS
- **F1 scores** — fixation and saccade F1 (only when `gt_labels.npy` is provided)
- **Downloadable CSV** — gaze samples with event classification ([see Output Reference](#output-reference))
- **Video overlay HTML** — the scene video with:
  - Gaze samples drawn as colored dots (fixation vs. saccade)
  - Fixation centers labeled with sequence numbers
  - An optical-flow arrow showing head motion
  - A clickable event timeline bar for seeking
  - Side-by-side comparison with ground truth (if provided)

---

## Recommended Parameters

In our paper *"Local Optical Flow for Eye Movement Event Detection in Head-Mounted Setups"*, we swept over a range of possible parameter values and reported the best one. The adaptive-threshold defaults in OpenGazeLab are the best values from our paper for Drews & Dierkes (2024) dataset ([OSF: 8en9v](https://osf.io/8en9v/overview), *"Strategies for enhancing automatic fixation detection in head-mounted eye tracking"*). 

### Adaptive-threshold defaults

| Parameter | Default | Justification |
|-----------|---------|---------------|
| `gain` | **0.05** | We swept gain over a wide range. **I-DT** performed best at `gain = 0`. **I-VT** performed best across `gain ∈ {0.4, 0.6, 0.7, 0.8}`. We chose **0.05** as a small, conservative compromise that nudges the threshold up under head motion without dominating either algorithm. |
| `window_size_ms` | **55.0** | Consistently the best window size across all parameter sweeps in the paper. |

### Detection-threshold suggestions

These values are the per-algorithm best detection thresholds we found on the Drews & Dierkes dataset:

| Algorithm | Suggested `detection_threshold` | Units |
|-----------|--------------------------------|-------|
| **I-DT** | **30** | relative-dispersion threshold in pixels |
| **I-VT** | **1** | relative-velocity threshold in px/ms |

For datasets recorded with a different headset, scene-camera resolution, or sampling rate, treat these as starting points and tune from there.

---

## Output Reference

The output CSV contains every input gaze sample plus these classification columns:

| Column | Description |
|--------|-------------|
| `x`, `y` | Gaze coordinates (pixels) |
| `timestamp` | Time in milliseconds |
| `event_type` | `Fixation`, `Saccade`, `NaN` (missing data), or `Out of Range Gaze Samples` |
| `fixation_x`, `fixation_y` | Fixation centroid coordinates (filled for fixation rows) |
| `fixation_id` | Unique fixation identifier |
| `saccade_id` | Unique saccade identifier |
| `event_duration` | Event duration in ms |
| `start_time`, `end_time` | Event temporal bounds in ms |

Head-mounted output additionally includes `flow_x`, `flow_y`, `video_timestamp`, `frame`, and `gt_label` (if ground truth was provided).

---

## Toolkit Structure

```
OpenGazeLab/
├── start_servers.bat                      # Windows one-click startup
├── README.md                              # This file
│
├── backend/                               # Python FastAPI server (port 5000)
│   ├── main.py                            # API endpoints (upload, plot, video streaming)
│   ├── requirements.txt                   # Python dependencies
│   └── src/
│       ├── __init__.py                    # Public exports (EventDetection, EyeTrackingVisualizer)
│       ├── pipeline.py                    # EventDetection — orchestrates the full pipeline
│       ├── algorithms.py                  # I-DT and I-VT classifiers
│       ├── feature_extraction.py          # I-VAT+Frel: Savgol smoothing, flow velocity, adaptive threshold
│       ├── preprocess_csv.py              # CSV parsing: delimiter/column/normalization auto-detection
│       ├── preprocess_headmounted.py      # Loader for .npy ZIP + scene video (Pupil Invisible format)
│       ├── utils.py                       # Velocity, MAD, fixation merging, timestamp helpers
│       └── visualization/
│           ├── __init__.py                # Visualization public exports
│           ├── stationary_plot.py         # Static Plotly: gaze + fixations + scanpath
│           ├── time_scrolling_plot.py     # Animated Plotly with playback controls
│           ├── video_overlay.py           # HTML5 video + canvas gaze overlay
│           └── _image_utils.py            # Encodes images as base64 for Plotly embedding
│   # data/ is created at runtime:
│   #   data/events/         — processed event CSVs (downloadable)
│   #   data/visualization/  — generated HTML visualizations and stored scene videos
│
└── frontend/                              # Static web UI (port 8000)
    ├── index.html                         # HTML entry point
    ├── package.json                       # Uses http-server (no build step)
    ├── package-lock.json
    └── src/
        ├── App.js                         # React app — mode toggle, upload forms, results display
        └── App.css                        # Styles
```

### What each backend file does

| File | Role |
|------|------|
| [main.py](backend/main.py) | FastAPI app. Defines endpoints `/api/upload`, `/api/upload-video`, `/api/plot/*`, `/api/plot-video/*`, `/api/video/*` |
| [pipeline.py](backend/src/pipeline.py) | `EventDetection` class — entry point that runs the full workflow: normalize → preprocess → detect → post-process |
| [algorithms.py](backend/src/algorithms.py) | The two core detection algorithms: `classify_idt` (dispersion) and `classify_ivt` (velocity) |
| [feature_extraction.py](backend/src/feature_extraction.py) | Head-mounted enhancements: Savitzky-Golay smoothing, flow velocity, relative velocity/dispersion, adaptive thresholds |
| [preprocess_csv.py](backend/src/preprocess_csv.py) | Reads CSV files: detects delimiter, column names, and coordinate normalization |
| [preprocess_headmounted.py](backend/src/preprocess_headmounted.py) | Reads `.npy` ZIP archives, aligns gaze with video frames |
| [utils.py](backend/src/utils.py) | Math helpers — velocity, MAD, fixation merging, timestamp normalization |
| [visualization/stationary_plot.py](backend/src/visualization/stationary_plot.py) | Builds the static Plotly chart |
| [visualization/time_scrolling_plot.py](backend/src/visualization/time_scrolling_plot.py) | Builds the animated playback Plotly chart |
| [visualization/video_overlay.py](backend/src/visualization/video_overlay.py) | Generates a self-contained HTML page with video + canvas gaze overlay |
| [visualization/_image_utils.py](backend/src/visualization/_image_utils.py) | Encodes background images as base64 data URIs for embedding in Plotly |

---

## How the Pipeline Works

End-to-end data flow for a single upload:

```
1. UPLOAD              Browser sends file(s) + parameters → FastAPI
2. PARSE               preprocess_csv.py / preprocess_headmounted.py
                       → DataFrame with x, y, timestamp (+ flow data for head-mounted)
3. NORMALIZE           pipeline.py — denormalize coords if needed,
                       convert timestamps to ms, separate invalid samples
4. FEATURE EXTRACTION  feature_extraction.py (head-mounted only)
                       → Savgol smoothing, gaze velocity, flow velocity,
                         relative velocity / relative dispersion, adaptive threshold
5. CLASSIFY            algorithms.py — I-DT or I-VT labels each sample
                       as Fixation or Saccade
6. POST-PROCESS        utils.py — merge nearby fixations, renumber IDs,
                       reinsert invalid samples with reason
7. VISUALIZE           visualization/ — generate Plotly HTML and/or
                       video-overlay HTML
8. RESPOND             Send statistics + visualization URLs to frontend
```

---

## Detection Algorithms Explained

### I-DT (Dispersion-Threshold Identification)
Classifies a window of samples as a **fixation** when their spatial dispersion stays below a threshold.
- **Dispersion formula**: `(max_x − min_x) + (max_y − min_y)` in pixels
- **Best for**: Low sampling rate, noisy data, stationary trackers
- **Typical threshold**: 100–200 pixels (stationary), ~30 px (head-mounted, on relative dispersion)

### I-VT (Velocity-Threshold Identification)
Classifies each sample as a **fixation** when point-to-point velocity stays below a threshold.
- **Velocity formula**: `sqrt(dx² + dy²) / dt` in pixels/ms
- **Best for**: High sampling rate, clean data
- **Typical threshold**: 20–50 px/ms (stationary), ~1.0 px/ms (head-mounted, on relative velocity)

### I-VAT+Frel (Head-Mounted Enhanced Pipeline)
A variant designed for head-mounted trackers, where head motion contaminates raw gaze velocity / dispersion. It feeds either the I-DT or the I-VT classifier with flow-compensated features:

1. **Savitzky-Golay smoothing** (55 ms window) on raw gaze coordinates
2. **Flow velocity** extracted from the optical flow grid (head/camera motion)
3. **For I-VT**: gaze velocity from smoothed coordinates, then **relative velocity** = `gaze_velocity − flow_velocity` — isolates true eye movement
4. **For I-DT**: **relative dispersion** — gaze dispersion measured against a flow-integrated "ideal" trajectory, removing apparent motion caused by head movement
5. **Flow RMS** (rolling 55 ms window) quantifies how much the head is moving
6. **Adaptive threshold** = `base + gain × flow_rms` — tightens during stillness, loosens during head movement
7. **Classification** compares the relative feature against the (adaptive or fixed) threshold

### Adaptive Thresholding
- **Stationary (MAD-based)**:
  ```
  adapted_threshold = original_threshold × (1 + tuning × MAD(velocity))
  ```
- **Head-mounted (flow-RMS-based)**:
  ```
  threshold_i = base_threshold + gain × flow_rms_i
  ```
  See [Recommended Parameters](#recommended-parameters) for the `gain` and `window_size_ms` defaults.

---

## Troubleshooting

### Backend won't start
- Ensure Python 3.11 or below is installed
- Install dependencies: `pip install -r backend/requirements.txt`
- Check whether port 5000 is in use; change it if needed
- Read the console for the actual error

### Frontend can't connect to backend
- Backend must be running on `http://127.0.0.1:5000`
- Frontend must be on `http://localhost:8000`
- Check the browser console for CORS errors
- Restart both servers

### Processing fails
- Check that the CSV has gaze columns and a timestamp (any of the supported names)
- Verify numeric columns don't contain text
- Try a different delimiter if auto-detection misfires
- Look for NaN values in coordinate columns

### No fixations detected
- Lower the detection threshold (see [Recommended Parameters](#recommended-parameters) for head-mounted starting points)
- Increase the minimum fixation duration if data is noisy
- Switch between I-DT and I-VT
- Enable adaptive thresholding
