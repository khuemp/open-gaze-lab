# OpenGazeLab

A web-based toolkit for processing eye-tracking gaze data and classifying it into **fixations** (eyes holding still on a target) and **saccades** (rapid eye movements between targets). Supports both **stationary eye trackers** and **head-mounted eye trackers**. Provides a Python processing pipeline and a browser-based UI for researchers working with eye-tracking data.

> **Note**: Python 3.10 recommended.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [How to Use — Stationary Eye Tracker](#how-to-use--stationary-eye-tracker)
3. [How to Use — Head-Mounted Eye Tracker](#how-to-use--head-mounted-eye-tracker)
4. [Input Reference](#input-reference)
5. [Recommended Parameters](#recommended-parameters)
6. [Output Reference](#output-reference)
7. [Toolkit Structure](#toolkit-structure)
8. [How the Pipeline Works](#how-the-pipeline-works)
9. [Detection Algorithms Explained](#detection-algorithms-explained)
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

The CSV needs three columns. Names and units are auto-detected as either pixels or normalized (0–1) for gaze, and milliseconds, seconds, or epoch for time(common variants supported, [see Input Reference](#input-reference))

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

1. **Dataset ZIP** (max 100 MB) — archive of dataset files
2. **Scene Camera Video** (max 5 GB) — `.mp4` from the head-mounted camera

OpenGazeLab auto-detects the dataset layout from the ZIP contents. Two layouts ship out of the box; bring your own data in either shape ([see Input Reference](#input-reference)).

#### Layout A — Drews & Dierkes (DD) style: `.npy` arrays

##### Required files inside the ZIP

| File | Shape | Description |
|------|-------|-------------|
| `gaze.npy` | (N, 2) | Eye gaze position (x, y) in pixels |
| `time_gaze.npy` | (N,) | Gaze timestamps in seconds |
| `optic_flow.npy` | (M, 11, 11, 2) | Per-frame 11×11 optical flow grid |
| `time_optic_flow.npy` | (M,) | Optical flow frame timestamps in seconds |
| `time_scene_camera.npy` | (M,) | Scene camera frame timestamps in seconds |

##### Optional files

| File | Shape | Description |
|------|-------|-------------|
| `gt_labels.npy` | (N,) | Ground truth labels (1 = Fixation, 0 = Saccade). Triggers automatic F1 score computation. |

`.npy` files may live at the ZIP root or inside a single subfolder.

#### Layout B — Gaze-in-Wild (GiW) style: `.mat` files + pre-computed flow

##### Required files inside the ZIP

| File | Description |
|------|-------------|
| `PrIdx_<P>_TrIdx_<T>.mat` | GiW signals file (exactly one). Filename participant/trial IDs are parsed to apply the labeler-priority rule. Provides gaze (`ProcessData.ETG.POR`), timestamps (`ProcessData.T`), and per-sample frame indices. |
| `PrIdx_<P>_TrIdx_<T>_Lbr_<N>.mat` | One or more labeler annotation files. |
| `optic_flow.npy` | `(M, 2)` per-frame mean optical flow |

##### How GiW labels are handled

GiW labels have six classes (UNDEFINED, FIXATION, PURSUIT, SACCADE, BLINK, FOLLOWING). OpenGazeLab collapses them to binary so the existing F1 scoring applies:

- **stable gaze (1)** = `FIXATION ∪ FOLLOWING` — i.e. eye-only fixation plus head+eye co-rotation tracking an attended target.
- **everything else (0)** = `SACCADE ∪ PURSUIT ∪ BLINK ∪ UNDEFINED`

Strict-fixation alone is rare in real-world recordings (~1% of samples on the trials we inspected), which is why following is lumped in.

When multiple labelers are present, OpenGazeLab uses the priority rule from Kothari et al.: trial 1 → labeler 5; otherwise prefer labeler 6, then 5, 1, 2, 3.

Triggers automatic F1 score computation just like a DD upload with `gt_labels.npy`.

Files may live at the ZIP root or inside any subfolder.

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

## Input Reference

OpenGazeLab can be used with both stationary and head-mounted eye-tracking datasets. The required input files and their formats differ between these two modes. Below is a reference for the example expected inputs in each case.

### Stationary Eye Tracker

| Reference |  |  |
|-----------|-------|---------|
| Disagreement Detection | [Paper](https://dl.acm.org/doi/pdf/10.1145/3772318.3790594) | [Dataset](https://github.com/DFKI-Interactive-Machine-Learning/Disagreement-Detection-Dataset-CHI-26) |
| gazeRE | [Paper](https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2021.808507/full) | [Dataset](https://github.com/DFKI-Interactive-Machine-Learning/gazeRE-dataset) |

### Head-Mounted Eye Tracker

| Reference |  |  |
|-----------|-------|---------|
| Drews & Dierkes (DD) | [Paper](https://link.springer.com/article/10.3758/s13428-024-02360-0) | [Dataset](https://osf.io/8en9v/overview) |
| Gaze-in-Wild (GiW) | [Paper](https://www.nature.com/articles/s41598-020-59251-5#Sec14) | [Dataset](https://www.cis.rit.edu/~rsk3900/gaze-in-wild/)  (Disclaimer: The website is no longer available, try this [repository](https://bitbucket.org/RSKothari/gaze-in-wild/src/master/) instead) |

---

## Recommended Parameters

### Adaptive-threshold suggestions

The adaptive-threshold path on the Head-Mounted tab takes two user inputs, `gain` and `window_size_ms`. Both default to `0` when left blank in the UI.

- **`gain`** — multiplier applied to the rolling RMS of optical-flow velocity. The per-sample threshold is computed as `detection_threshold + gain × flow_rms_mag`, so a larger `gain` pushes the threshold up more aggressively under head motion. With `gain = 0`, the motion-driven adjustment is disabled (the threshold equals `detection_threshold` for every sample).
- **`window_size_ms`** — length of the centered rolling window used to compute the flow-RMS magnitude. Shorter windows track rapid head movements more closely; longer windows produce a smoother, less reactive threshold.

The recommendations below come from our parameter sweeps on the DD and GiW datasets listed in [Input Reference](#input-reference). Treat them as practical starting points — tune for recordings on different headsets, scene cameras, or sampling rates.

| Dataset | Algorithm | `gain`                 | `window_size_ms` |
|---------|-----------|------------------------|-------------------|
| DD      | I-DT      | ∈ {0.4, 0.6, 0.7, 0.8} | 55                |
| DD      | I-VT      | 0                      | 55                |
| GiW     | I-DT      | 0.05                   | 155               |
| GiW     | I-VT      | ∈ {0.6, 0.8, 0.9, 1.0} | 55                |

### Detection-threshold suggestions

These values are the per-algorithm best detection thresholds found on the DD dataset:

| Algorithm |`detection_threshold` | Units |
|-----------|--------------------------------|-------|
| **I-DT** | **30** | relative-dispersion threshold in pixels |
| **I-VT** | **1.5** | relative-velocity threshold in px/ms |

Tune these values for datasets recorded with a different headset, scene-camera resolution, or sampling rate.

### DD parameters

| Parameter | Suggested value | 
|-----------|-----------------|
| Video Resolution | `1088,1080` | 
| Sampling Rate | `200` | 

### GiW parameters

| Parameter | Suggested value |
|-----------|-----------------|
| Video Resolution | `1920,1080` | 
| Sampling Rate | `300` | 

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
│       ├── feature_extraction.py          # Head-mounted pipeline: Savgol smoothing, flow velocity, adaptive threshold
│       ├── preprocess_csv.py              # CSV parsing: delimiter/column/normalization auto-detection
│       ├── preprocess_headmounted/        # Head-mounted loader package
│       │   ├── __init__.py                # Re-exports the public API
│       │   ├── common.py                  # Shared helpers (extract_video_metadata)
│       │   ├── dd.py                      # DD (.npy) loader
│       │   ├── giw.py                     # GiW (.mat) loader
│       │   └── dispatcher.py              # Auto-routes DD vs GiW by ZIP contents
│       ├── utils.py                       # Velocity, MAD, fixation merging, timestamp helpers
│       └── visualization/
│           ├── __init__.py                # Visualization public exports
│           ├── stationary_plot.py         # Static Plotly: gaze + fixations + scanpath
│           ├── time_scrolling_plot.py     # Animated Plotly with playback controls
│           ├── video_overlay.py           # HTML5 video + canvas gaze overlay
│           └── _image_utils.py            # Encodes images as base64 for Plotly embedding
│   └── data/                              # Created at runtime
│       ├── events/                        # Processed event CSVs (downloadable)
│       └── visualization/                 # Generated HTML visualizations and stored scene videos
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
| [feature_extraction.py](backend/src/feature_extraction.py) | Head-mounted pipeline: Savitzky-Golay smoothing, flow velocity, relative velocity/dispersion, adaptive thresholds |
| [preprocess_csv.py](backend/src/preprocess_csv.py) | Reads CSV files: detects delimiter, column names, and coordinate normalization |
| [preprocess_headmounted/](backend/src/preprocess_headmounted/) | Head-mounted loader package. Contains `dd.py` (Drews `.npy` loader), `giw.py` (Gaze-in-Wild `.mat` loader), `common.py` (shared video-metadata helper), and `dispatcher.py` (auto-routes uploads by inspecting the ZIP — any `.mat` entry → GiW, otherwise DD). |
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
2. PARSE               preprocess_csv.py / preprocess_headmounted/ (auto-routes DD vs GiW) → DataFrame with x, y, timestamp (+ flow data for head-mounted)
3. NORMALIZE           pipeline.py — denormalize coords if needed, convert timestamps to ms, separate invalid samples
4. FEATURE EXTRACTION  feature_extraction.py (head-mounted only) → Savgol smoothing, gaze velocity, flow velocity, relative velocity / relative dispersion, adaptive threshold
5. CLASSIFY            algorithms.py — I-DT or I-VT labels each sample as Fixation or Saccade
6. POST-PROCESS        utils.py — merge nearby fixations, renumber IDs, reinsert invalid samples with reason
7. VISUALIZE           visualization/ — generate Plotly HTML and/or video-overlay HTML
8. RESPOND             Send statistics + visualization URLs to frontend
```

---

## Detection Algorithms Explained

### I-DT (Dispersion-Threshold Identification)
Classifies a window of samples as a **fixation** when their spatial dispersion stays below a threshold.
- **Dispersion formula**: `(max_x − min_x) + (max_y − min_y)` in pixels
- **Best for**: Low sampling rate, noisy data, stationary trackers

### I-VT (Velocity-Threshold Identification)
Classifies each sample as a **fixation** when point-to-point velocity stays below a threshold.
- **Velocity formula**: `sqrt(dx² + dy²) / dt` in pixels/ms
- **Best for**: High sampling rate, clean data

### Head-Mounted Enhanced Pipeline
A variant designed for head-mounted trackers, where head motion contaminates raw gaze velocity / dispersion. It feeds either the I-DT or the I-VT classifier with flow-compensated features:

1. **Savitzky-Golay smoothing** (55 ms window, 3rd order) on raw gaze coordinates
2. **Flow velocity** extracted from the optical flow grid (head/camera motion)
3. **For I-VT**: gaze velocity from smoothed coordinates, then **relative velocity** = `gaze_velocity − flow_velocity` — isolates true eye movement
4. **For I-DT**: **relative dispersion** — gaze dispersion measured against a flow-integrated "ideal" trajectory, removing apparent motion caused by head movement
5. **Flow RMS** = `sqrt(mean(flow_x_vel²) + mean(flow_y_vel²))` where `flow_x_vel, flow_y_vel = flow_x, flow_y / flow_t_delta` — quantifies how much the head is moving
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
- Ensure Python 3.10 or below is installed
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
