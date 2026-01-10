# Event Detection Frontend

Simple React frontend (no build tools needed) for eye tracking fixation detection.

## How to Run

**Option 1: Using Python's built-in server**
```bash
cd event_detection-frontend
python -m http.server 8000
```
Then open `http://localhost:8000`

**Option 2: Using Node.js http-server**
```bash
npm install -g http-server
http-server
```

**Option 3: Direct file access**
Simply open `index.html` in your browser

## Project Structure

```
event_detection-frontend/
├── index.html          # Main HTML file
├── src/
│   ├── App.js         # React app (all components in one file)
│   ├── App.css        # Application styles
│   └── index.css      # Global styles
└── README.md
```

## Features

✅ **No build process required** - uses React from CDN  
✅ **CSV file upload** with validation  
✅ **Threshold slider** (0-1) with live value display  
✅ **Quick presets** (Sensitive, Medium, Strict)  
✅ **Results display** with metrics  
✅ **Responsive design** (mobile/tablet friendly)  
✅ **Dark theme** with modern UI  

## API Integration

The frontend expects a backend at `http://localhost:8000/api/process` with:
- `POST /process` endpoint accepting `file` (CSV) and `threshold` (float)
- Response: JSON with `fixation_count`, `processing_time`, `total_points`, etc.

## File Size

- No node_modules
- No build step
- ~5KB total JavaScript
- **Super lightweight!**
