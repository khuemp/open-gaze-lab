# Quick Start Guide

## 1. Install Dependencies

### Backend
```bash
cd event_detection-backend
pip install -r requirements.txt
```

### Frontend
```bash
cd event_detection-frontend
npm install
```

## 2. Start the Servers

### Option A: Windows Batch Script (Easiest)
Double-click `start_servers.bat` in the root directory. This will open two command windows and start both servers automatically.

### Option B: Manual Start (Any OS)

**Terminal 1 - Backend:**
```bash
cd event_detection-backend
python app.py
```
You should see:
```
[INFO] Starting Flask server...
 * Running on http://127.0.0.1:5000
```

**Terminal 2 - Frontend:**
```bash
cd event_detection-frontend
npm start
```
Browser should open at `http://localhost:8000`

## 3. Use the Application

1. **Open** `http://localhost:8000` (should auto-open with npm start)

2. **Select a CSV file** containing gaze data with columns like:
   - `x`, `y` (required)
   - `timestamp` (optional)

3. **Configure parameters** (or use defaults):
   - Display Resolution: 2560,1440
   - Minimal Fixation Duration: 50 ms
   - Detection Threshold: 0.5
   - Algorithm: I-DT
   - Sampling Rate: 1000 Hz

4. **Click "Process Gaze Data"** and wait for processing

5. **View results**:
   - See statistics (fixations, saccades, total events)
   - Click **"Download Events CSV"** to get the processed data
   - Click **"View Plot"** to see interactive visualization

## Example Test Data

Create a simple `test.csv`:
```
timestamp;x;y
1000;1280;720
1010;1281;720
1020;1282;720
1030;1283;720
1040;1284;720
1050;1285;720
1060;1286;720
1070;1287;720
1080;1288;720
1090;1289;720
```

Upload this and try with default parameters.

## Troubleshooting

### "Backend not responding" error
- [ ] Check Terminal 1 - Flask server should show "Running on http://127.0.0.1:5000"
- [ ] Try opening http://127.0.0.1:5000/api/status in browser - should return `{"status":"ok"}`
- [ ] Wait 5 seconds after starting Flask before opening frontend

### "CORS error" or blocked requests
- [ ] Backend must be running on `http://127.0.0.1:5000` (not 127.0.0.1:5000/)
- [ ] Frontend must be on `http://localhost:8000` (not 127.0.0.1:8000)
- [ ] Restart both servers if you changed ports

### "No module named 'flask'"
```bash
cd event_detection-backend
pip install -r requirements.txt
```

### "npm: command not found"
- [ ] Install Node.js from https://nodejs.org/
- [ ] Restart your terminal after installation
- [ ] Run `npm --version` to verify

### Results show 0 events
- [ ] Check CSV format - must have `x` and `y` columns
- [ ] Check delimiter is auto-detected (semicolon, comma, tab, or space)
- [ ] Try adjusting detection threshold or algorithm

## Key Files

| File | Purpose |
|------|---------|
| `event_detection-backend/app.py` | Flask API server |
| `event_detection-frontend/src/App.js` | React frontend |
| `event_detection-frontend/src/index.css` | Frontend styling |
| `README.md` | Full project documentation |
| `MODIFICATIONS.md` | What was changed |

## Tips

- Use smaller CSV files (< 1MB) for testing
- Processing time depends on file size (typically 5-30 seconds)
- Results are saved with timestamps, so you can upload multiple files
- Check browser console (F12) for detailed error messages
- Check terminal output for backend processing logs

## Next Steps

- Read `README.md` for detailed API documentation
- Check `MODIFICATIONS.md` for technical details
- Review `event_detection-backend/app.py` for API implementation
- Review `event_detection-frontend/src/App.js` for frontend logic
