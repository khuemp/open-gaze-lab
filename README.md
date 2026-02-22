# Eye-tracking Event Detection Toolkit

A web-based application for processing eye-tracking gaze data to detect and classify **fixations**, **saccades**, and **blinks**. Provides both a Python processing pipeline and a modern web interface for researchers working with eye-tracking data.

> **Note**: Requires Python 3.11 or below.

---

## Features

### Data Processing
- **Automatic delimiter detection**: Supports `;`, `,`, `\t`, `|`, and space
- **Automatic column mapping**: Detects x/y/timestamp column names automatically
- **Coordinate normalization detection**: Auto-detects normalized (0-1) vs pixel coordinates
- **Timestamp correction**: Regularizes irregular timestamps to uniform intervals
- **Blink detection**: Identifies NaN/missing gaze data as blinks
- **Fixation merging**: Combines spatially close consecutive fixations using duration-weighted averaging

### Detection Algorithms
- **I-DT (Dispersion-Threshold)**: Classifies fixations based on spatial dispersion
- **I-VT (Velocity-Threshold)**: Classifies fixations based on point-to-point velocity
- **Adaptive thresholding**: Adjusts threshold based on movement variability (MAD)
- **Threshold optimization**: Uses Calinski-Harabasz score for automatic parameter tuning

### Visualization
- **Static visualization**: Interactive Plotly plot with gaze points, fixations, and scanpath
- **Time-scrolling visualization**: Animated playback with play/pause controls and time slider
- **AOI overlay**: Display Areas of Interest on visualizations
- **Background image support**: Overlay stimulus screen image behind gaze plots

### Web Interface
- **Drag-and-drop upload**: Easy CSV and background image selection
- **Parameter configuration**: Full control over detection parameters
- **Real-time results**: Statistics display (fixations, saccades, blinks counts)
- **CSV export**: Download processed event data
- **Embedded visualizations**: View plots directly in browser

---

## Quick Start

### Option A: Windows (Easiest)
Double-click `start_servers.bat` in the root directory. This opens two terminals and starts both servers automatically.

### Option B: Manual Start

**1. Start Backend:**
```bash
cd event_detection-backend
pip install -r requirements.txt
python main.py
```
Backend runs at: `http://127.0.0.1:5000`

**2. Start Frontend:**
```bash
cd event_detection-frontend
npm install
npm run start
```
Frontend opens at: `http://localhost:8000`

---

## Project Structure

```
fixation_detection/
├── start_servers.bat              # Windows startup script
├── README.md                      # This file
├── GUIDE.md                       # Quick start guide
│
├── event_detection-backend/       # Python FastAPI backend
│   ├── main.py                    # API server
│   ├── requirements.txt           # Python dependencies
│   ├── src/
│   │   ├── __init__.py            # EventDetection & EyeTrackingVisualizer classes
│   │   ├── detection_algorithms.py # I-DT and I-VT implementations
│   │   ├── pipeline.py            # Event detection pipeline
│   │   ├── visualization.py       # Plotly visualizations
│   │   ├── utils.py               # Velocity, MAD, merging utilities
│   │   └── aoi.py                 # AOI classification
│   └── data/
│       ├── uploads/               # Uploaded CSV files
│       ├── images/                # Uploaded background images
│       ├── events/                # Processed event CSVs
│       └── visualization/         # HTML visualizations
│
└── event_detection-frontend/      # React frontend
    ├── index.html
    ├── package.json
    └── src/
        ├── App.js                 # Main React application
        ├── App.css                # Component styles
        └── index.css              # Global styles
```

---

## Usage

### Step 1: Upload CSV
Click or drag-and-drop your gaze data CSV file into the upload area.

### Step 1b: Upload Background Image (Optional)
Optionally upload a screenshot or stimulus image (PNG, JPG, BMP, GIF, WebP) to display behind the gaze plots. The image is mapped to the full display resolution so gaze positions align with on-screen content. When a background image is present, the plot grid is hidden and the background is made transparent for a cleaner overlay.

### Step 2: Configure Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| **Display Resolution** | 2560,1440 | Screen resolution in pixels (wI-DTh,height) |
| **Minimal Fixation Duration** | 100 ms | Minimum duration for valid fixation |
| **Detection Threshold** | 125 | For I-DT: dispersion in pixels. For I-VT: velocity in pixels/ms |
| **Algorithm** | I-DT | Detection algorithm: `I-DT` or `I-VT` |
| **Sampling Rate** | 250 Hz | Eye-tracker sampling rate |
| **Merge Threshold** | None | Max distance (px) to merge nearby fixations |
| **Adaptive Threshold** | Off | Enable adaptive threshold adjustment |

### Step 3: Process
Click **"Process Gaze Data"** and wait for processing.

### Step 4: View Results
- **Statistics**: Total events, fixation points, saccade points, blink points
- **Download CSV**: Get processed event data
- **Static Plot**: View gaze points, fixations, and scanpath
- **Time-Scrolling Plot**: Watch fixations appear over time with playback controls

---

## Detection Algorithms Explained

### I-DT (Dispersion-Threshold Identification)
Classifies gaze points as fixations when **spatial dispersion** within a temporal window is below a threshold.

- **Dispersion formula**: `(max_x - min_x) + (max_y - min_y)` in pixels
- **Best for**: Low sampling rate data, noisy data
- **Typical threshold**: 100-200 pixels

### I-VT (Velocity-Threshold Identification)
Classifies gaze points as fixations when **point-to-point velocity** is below a threshold.

- **Velocity formula**: `sqrt(dx² + dy²) / dt` in pixels/ms
- **Best for**: High sampling rate data, clean data
- **Typical threshold**: 20-50 pixels/ms

### Adaptive Thresholding
When enabled, the threshold is automatically adjusted based on movement variability:
```
adapted_threshold = original_threshold × (1 + tuning_parameter × MAD(velocity))
```
This helps handle individual differences in eye movement patterns.

---

## Input CSV Format

Your CSV file should contain columns for gaze coordinates and timestamps:

| Column | Required | Description |
|--------|----------|-------------|
| `x` or `gaze_x` | Yes | X coordinate (pixels or normalized 0-1) |
| `y` or `gaze_y` | Yes | Y coordinate (pixels or normalized 0-1) |
| `timestamp` or `time` | Yes | Time in milliseconds or seconds |

**Auto-detected delimiters**: `;`, `,`, `\t`, `|`, space

### Example CSV
```csv
timestamp;x;y
1000;1280;720
1010;1281;720
1020;1282;720
1030;1500;800
1040;1501;801
```

---

## Output Data Structure

Each gaze point in the output CSV is classified with:

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
