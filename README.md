# Eye Tracking Fixation Detection - Web Application

A web application for uploading eye-tracking gaze data and detecting fixations with customizable parameters.

**New Feature**: ⏱️ Time-scrolling visualization! See how fixations and gaze points appear progressively over time with interactive play/pause controls and a time slider. [Learn more](TIME_SCROLLING_GUIDE.md)

Not support python versions above 3.11 yet.

## Project Structure

```
event_detection-backend/     # FastAPI backend
  ├── app.py                 # FastAPI application
  ├── src/                   # Event detection modules
  │   ├── pipeline.py        # EventDetection pipeline
  │   ├── detection_algorithms.py
  │   ├── visualization.py   # Visualization functions
  │   └── ...
  └── data/                  # Output directories
      ├── uploads/           # Uploaded CSV files
      └── results/           # Processing results

event_detection-frontend/    # React frontend
  ├── index.html             # Main HTML file
  ├── package.json           # NPM dependencies
  └── src/
      ├── App.js             # Main React component
      ├── index.css          # Styling
      └── App.css            # Component styles
```

## Installation

### Backend

1. Install Python dependencies:
```bash
cd event_detection-backend
pip install -r requirements.txt
```

2. Start the FastAPI server:
```bash
python app.py
```

The backend will run on `http://127.0.0.1:5000`

**Interactive API Documentation**: Visit `http://127.0.0.1:5000/docs` for Swagger UI

### Frontend

1. Install Node dependencies:
```bash
cd event_detection-frontend
npm install
```

2. Start the development server:
```bash
npm start
```

The frontend will open at `http://localhost:8000`

## Usage

1. **Upload CSV**: Click on the upload area to select your gaze data CSV file
2. **Configure Parameters**:
   - **Display Resolution**: Screen resolution in WxH format (e.g., 2560,1440)
   - **Min Fixation Duration**: Minimum duration for a fixation in milliseconds (default: 50)
   - **Detection Threshold**: Sensitivity for fixation detection (0.0-1.0, default: 0.5)
   - **Algorithm**: Choose between IDT or IVT detection algorithm
   - **Sampling Rate**: Eye-tracker sampling rate in Hz (default: 1000)
3. **Process**: Click the "Process Gaze Data" button
4. **View Results**: Once processing completes, you can:
   - View summary statistics (number of fixations, saccades, events)
   - Download the events CSV file
   - View interactive plot of gaze data

## API Endpoints

### `POST /api/upload`
Upload CSV file with detection parameters.

**Form Data:**
- `file` (file): CSV file containing gaze data
- `resolution` (string): Display resolution "width,height"
- `min_fixation_duration` (int): Minimum fixation duration in ms
- `detect_threshold` (float): Detection threshold (0-1)
- `algorithm` (string): Detection algorithm ('idt' or 'ivt')
- `sampling_rate` (int): Sampling rate in Hz

**Response:**
```json
{
  "success": true,
  "message": "File processed successfully",
  "filename": "20240111_120530",
  "result": {
    "num_events": 1250,
    "num_fixations": 150,
    "num_saccades": 100,
    "events_file": "data/results/...",
    "plot_file": "data/results/gaze_plot.html"
  }
}
```

### `GET /api/results/<filename>`
Download the events CSV file for a processed dataset.

### `GET /api/plot/<filename>`
Get the interactive HTML plot for a processed dataset.

### `GET /api/status`
Health check endpoint.

## CSV Format

Your gaze data CSV should contain the following columns:
- `x`: X coordinate of gaze point
- `y`: Y coordinate of gaze point
- `timestamp` (optional): Timestamp of the gaze point

The backend automatically detects common delimiters: `;`, `,`, `\t`, and space.

## Example CSV

```
timestamp;x;y
1000;1280;720
1010;1281;721
1020;1282;722
```

## Features

- **Automatic Delimiter Detection**: Works with semicolon, comma, tab, or space-separated CSV files
- **Flexible Parameters**: Customize detection algorithm, sensitivity, and resolution
- **Real-time Processing**: Process large gaze datasets on the fly
- **Interactive Visualization**: View gaze points, fixations, and saccades with Plotly
- **Data Export**: Download processed events as CSV for further analysis

## Troubleshooting

### Backend won't start
- Check if FastAPI and uvicorn are installed: `pip install -r requirements.txt`
- Make sure port 5000 is not in use
- Check console output for error messages

### Frontend can't connect to backend
- Ensure backend is running on `http://127.0.0.1:5000`
- Visit `http://127.0.0.1:5000/docs` to verify API is working
- Check browser console for CORS errors
- Frontend should be on `http://localhost:8000`

### Processing fails
- Check CSV format (expected columns: x, y, and optionally timestamp)
- Verify delimiter is correctly auto-detected
- Check console logs for specific error messages

## Development

To modify the application:
- **Backend**: Edit files in `event_detection-backend/src/`
- **Frontend**: Edit files in `event_detection-frontend/src/`

After making changes, restart the respective servers.
