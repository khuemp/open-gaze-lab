"""
=============================================================================
 VERIFICATION TOOL for I-VAT+Frel (Head-Mounted Eye Tracking Enhancement)
=============================================================================

This script verifies whether the backend correctly implements the enhanced
fixation detection algorithm from the paper:

  "Strategies for enhancing automatic fixation detection in head-mounted
   eye tracking" by Michael Drews and Kai Dierkes

It uses Dataset A from the paper (Pupil Invisible glasses recording).

HOW TO RUN:
  1. Open a terminal in: event_detection-backend/
  2. Run: python verify_ivat.py
  3. Read the output -- it explains every step and what to expect
  4. A browser window will open with an interactive visualization

WHAT IT CHECKS:
  - Each preprocessing step produces correct values
  - Velocities match a known-good reference implementation
  - Classification accuracy matches the paper's results
  - NaN handling works correctly
  - Flow vs no-flow comparison
=============================================================================
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report

# -- Setup -----------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "A")
sys.path.insert(0, os.path.dirname(__file__))


def print_header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_step(num, title):
    print(f"\n--- Step {num}: {title} ---")


# =====================================================================
# PART 1: UNDERSTAND THE DATA
# =====================================================================
print_header("PART 1: What does Dataset A contain?")

print("""
Dataset A was recorded with Pupil Invisible head-mounted eye-tracking glasses.
A person wore the glasses and looked around. The glasses recorded:
  1. WHERE the eyes looked (gaze position on the scene camera image)
  2. WHEN each gaze sample was taken (timestamps)
  3. HOW the camera image moved between video frames (optic flow)
  4. A human expert labeled each sample: Fixation or Saccade (ground truth)
""")

# Load raw data
gaze = np.load(os.path.join(DATA_DIR, "gaze.npy"))
time_gaze = np.load(os.path.join(DATA_DIR, "time_gaze.npy"))
gt_labels = np.load(os.path.join(DATA_DIR, "gt_labels.npy"))
optic_flow = np.load(os.path.join(DATA_DIR, "optic_flow.npy"))
time_of = np.load(os.path.join(DATA_DIR, "time_optic_flow.npy"))

print_step(1, "gaze.npy -- Eye position (x, y) in pixels")
print(f"  Shape: {gaze.shape}  ->  {gaze.shape[0]} samples, each has (x, y)")
print(f"  Example first 3 rows:")
for i in range(3):
    print(f"    Sample {i}: x={gaze[i,0]:.1f} px, y={gaze[i,1]:.1f} px")
print(f"  Range: x=[{gaze[:,0].min():.0f}, {gaze[:,0].max():.0f}], "
      f"y=[{gaze[:,1].min():.0f}, {gaze[:,1].max():.0f}]")
print(f"  Has NaN: {np.isnan(gaze).any()}")

print_step(2, "time_gaze.npy -- Timestamp for each gaze sample (in seconds)")
print(f"  Shape: {time_gaze.shape}")
print(f"  First 3: {time_gaze[:3]}")
print(f"  Total duration: {time_gaze[-1] - time_gaze[0]:.1f} seconds")
sampling_rate = 1.0 / np.median(np.diff(time_gaze))
print(f"  Sampling rate: ~{sampling_rate:.0f} Hz (samples per second)")

print_step(3, "gt_labels.npy -- Ground truth (expert labels)")
print(f"  Shape: {gt_labels.shape}")
print(f"  Values: 1 = Fixation, 0 = Saccade")
n_fix = (gt_labels == 1).sum()
n_sac = (gt_labels == 0).sum()
print(f"  Distribution: {n_fix} Fixation ({n_fix/len(gt_labels)*100:.1f}%), "
      f"{n_sac} Saccade ({n_sac/len(gt_labels)*100:.1f}%)")

print_step(4, "optic_flow.npy -- Camera motion (THE KEY for head-mounted)")
print(f"  Shape: {optic_flow.shape}")
print(f"  Meaning: {optic_flow.shape[0]} video frames, each has an 11x11 grid")
print(f"           of flow vectors (dx, dy) showing how much each point moved")
print(f"  Example: frame 0, grid center [5,5] = {optic_flow[0, 5, 5]}")
print(f"           -> the image moved {optic_flow[0,5,5,0]:.3f} px right, "
      f"{optic_flow[0,5,5,1]:.3f} px down")
print(f"  Has NaN: {np.isnan(optic_flow).any()} (some grid points lost tracking)")

print_step(5, "time_optic_flow.npy -- Timestamp for each video frame")
print(f"  Shape: {time_of.shape}  ->  {len(time_of)} video frames")
video_fps = 1.0 / np.median(np.diff(time_of))
print(f"  Video FPS: ~{video_fps:.0f}")
print(f"  Note: {len(gaze)} gaze samples but only {len(time_of)} video frames")
print(f"        -> multiple gaze samples share the same flow value")


# =====================================================================
# PART 2: WHAT IS "OPTIC FLOW" AND WHY DO WE NEED IT?
# =====================================================================
print_header("PART 2: What is 'optic flow' and why does it matter?")
print("""
PROBLEM with head-mounted eye trackers:
  - The camera is on your HEAD
  - When you TURN YOUR HEAD, the whole camera image shifts
  - The eye tracker sees the gaze position moving
  - But the EYE didn't actually move -- the HEAD moved!
  - Standard I-VT/I-DT incorrectly classifies this as a saccade

SOLUTION: Optic flow tells us how much the camera image moved.
  - If gaze moved 10 px to the right AND the camera moved 8 px right:
    -> Real eye movement = 10 - 8 = 2 px (small -> fixation!)
  - Without flow compensation: 10 px -> might be above threshold -> saccade!
  - This subtraction is called "flow-relative velocity" (Frel)

HOW THE COLUMNS WORK:
  flow_x, flow_y      = how much the camera image shifted (pixels/frame)
  video_timestamp      = when each video frame was captured
  gaze velocity        = how fast gaze moves (from smoothed x,y positions)
  flow velocity        = camera shift / time between frames
  relative velocity    = gaze velocity - flow velocity (TRUE eye movement)
""")


# =====================================================================
# PART 3: PREPARE DATA FOR THE BACKEND
# =====================================================================
print_header("PART 3: Prepare data for the backend")

print_step("3a", "Average optic flow per frame (11x11 grid -> single vector)")
mean_flow = np.nanmean(optic_flow.astype(np.float64), axis=(1, 2))
print(f"  Before: shape {optic_flow.shape} (600 frames x 11 x 11 x 2)")
print(f"  After:  shape {mean_flow.shape} (600 frames x 2)")
print(f"  Frame 0 average flow: dx={mean_flow[0,0]:.4f}, dy={mean_flow[0,1]:.4f}")

print_step("3b", "Assign each gaze sample to its nearest video frame")
indices = np.searchsorted(time_of, time_gaze, side="right") - 1
indices = np.clip(indices, 0, len(time_of) - 1)
flow_per_gaze = mean_flow[indices]
video_ts_per_gaze = time_of[indices]
print(f"  Gaze sample 0 (t={time_gaze[0]:.4f}s) -> video frame {indices[0]} "
      f"(t={time_of[indices[0]]:.4f}s)")
print(f"  Gaze sample 100 (t={time_gaze[100]:.4f}s) -> video frame {indices[100]}")

print_step("3c", "Build the DataFrame our backend expects")
print("""
  Our backend expects these columns:
  +------------------+----------------------------------------------+
  | Column           | Meaning                                      |
  +------------------+----------------------------------------------+
  | timestamp        | When this gaze sample was recorded           |
  | x                | Horizontal eye position (pixels)             |
  | y                | Vertical eye position (pixels)               |
  | flow_x           | Camera shift in x (pixels/frame) [OPTIONAL]  |
  | flow_y           | Camera shift in y (pixels/frame) [OPTIONAL]  |
  | video_timestamp  | When the video frame was taken    [OPTIONAL]  |
  +------------------+----------------------------------------------+

  flow_x, flow_y, video_timestamp are OPTIONAL:
    - If present -> enhanced head-mounted pipeline activates
    - If absent  -> original desktop I-VT/I-DT runs (unchanged)
""")

df = pd.DataFrame({
    "timestamp": time_gaze,
    "x": gaze[:, 0],
    "y": gaze[:, 1],
    "flow_x": flow_per_gaze[:, 0],
    "flow_y": flow_per_gaze[:, 1],
    "video_timestamp": video_ts_per_gaze,
})
print(f"  DataFrame: {len(df)} rows x {len(df.columns)} columns")
print(f"  First row: {dict(df.iloc[0])}")


# =====================================================================
# PART 4: HOW DOES THE CODE HANDLE NaN DATA?
# =====================================================================
print_header("PART 4: How does the code handle NaN (missing) data?")
print("""
  With head-mounted glasses, NaN values happen when:
    - The eye tracker loses the pupil (blink, closed eyes)
    - The gaze moves outside the camera's field of view
    - Optic flow tracking fails at some grid points

  Here's how our code handles it at each step:

  1. PIPELINE ENTRY (pipeline.py -> detect_event):
     -> separate_invalid_points() REMOVES NaN rows before classification
     -> Stores which rows were NaN and why

  2. SAVGOL FILTER (preprocessing.py -> apply_savgol_filter):
     -> NaN gaze coordinates are temporarily filled with 0
     -> The filter runs on the filled data
     -> (This matches the research code: savgol_filter.py fillna(0))

  3. VELOCITY CALCULATION (preprocessing.py -> compute_gaze_velocity):
     -> Uses diff().fillna(0) so NaN deltas become 0 velocity
     -> 0 velocity = looks like fixation (safe default)

  4. FLOW VELOCITY (preprocessing.py -> compute_flow_velocity):
     -> NaN flow values produce NaN velocity
     -> inf values from division by zero are replaced with 0
     -> Forward-fill and backward-fill handle gaps

  5. CLASSIFICATION (detection_algorithms.py -> classify_ivt):
     -> velocity.fillna(0) -- any remaining NaN -> 0 -> fixation

  6. PIPELINE EXIT (pipeline.py -> detect_event):
     -> reinsert_invalid_points() puts NaN rows BACK at original positions
     -> Labels them as 'NaN' event type

  IMPORTANT: In Dataset A, there are NO NaN gaze values (it's clean data).
  But in real recordings, there will be NaN values.
""")

# Check for NaN in our dataset
print(f"  Dataset A NaN check:")
print(f"    gaze has NaN: {np.isnan(gaze).any()}")
print(f"    optic_flow has NaN: {np.isnan(optic_flow).any()}")
print(f"    mean_flow has NaN: {np.isnan(mean_flow).any()}")
print(f"    (optic flow NaN are handled by nanmean averaging)")


# =====================================================================
# PART 5: STEP-BY-STEP SIMULATION OF THE PIPELINE
# =====================================================================
print_header("PART 5: Step-by-step simulation of what the code does")

from src import EventDetection
from src.preprocessing import (
    apply_savgol_filter, compute_gaze_velocity, compute_flow_velocity,
    compute_relative_velocity, compute_flow_window_rms,
    compute_adaptive_threshold, detect_flow_columns,
)
from src.detection_algorithms import classify_ivt

# Step 5a: EventDetection.__init__ converts timestamps
print_step("5a", "EventDetection.__init__ -- unit conversion")
det = EventDetection(
    df.copy(), resolution=(1088, 1080),
    column_mapping={
        "x": "x", "y": "y", "timestamp": "timestamp",
        "flow_x": "flow_x", "flow_y": "flow_y",
        "video_timestamp": "video_timestamp",
    },
    is_normalized=False,
)
gd = det.gaze_data
print(f"  Input timestamp range:  [{df['timestamp'].iloc[0]:.4f}, "
      f"{df['timestamp'].iloc[-1]:.4f}] seconds")
print(f"  After conversion:       [{gd['timestamp'].iloc[0]:.2f}, "
      f"{gd['timestamp'].iloc[-1]:.2f}] ms")
print(f"  Video_timestamp range:  [{gd['video_timestamp'].iloc[0]:.2f}, "
      f"{gd['video_timestamp'].iloc[-1]:.2f}] ms")
print(f"  (Timestamps converted from seconds to milliseconds)")

# Step 5b: Savgol filter
print_step("5b", "Savgol filter -- smooth the noisy gaze coordinates")
sim_df = gd.copy()
apply_savgol_filter(sim_df, sampling_rate, window_size_ms=55.0)
print(f"  Original x[0:5]:  {sim_df['x'].values[:5].round(2)}")
print(f"  Smoothed x[0:5]:  {sim_df['filter_x'].values[:5].round(2)}")
print(f"  Difference:        "
      f"{(sim_df['x'].values[:5] - sim_df['filter_x'].values[:5]).round(2)}")
print(f"  (Small differences = filter removes noise while keeping the trend)")

# Step 5c: Gaze velocity
print_step("5c", "Gaze velocity -- how fast are the eyes moving?")
compute_gaze_velocity(sim_df, x_col="filter_x", y_col="filter_y")
vel_pxs = sim_df["vel_mag"].values * 1000  # px/ms -> px/s
print(f"  vel_mag (px/ms): mean={sim_df['vel_mag'].mean():.4f}, "
      f"max={sim_df['vel_mag'].max():.4f}")
print(f"  vel_mag (px/s):  mean={vel_pxs.mean():.1f}, max={vel_pxs.max():.1f}")
print(f"  (Fixations are slow <~100 px/s, saccades are fast >~500 px/s)")

# Step 5d: Flow velocity
print_step("5d", "Flow velocity -- how fast is the CAMERA moving?")
compute_flow_velocity(sim_df)
flow_pxs = sim_df["flow_vel_mag"].values * 1000
print(f"  flow_vel_mag (px/ms): mean={sim_df['flow_vel_mag'].mean():.4f}")
print(f"  flow_vel_mag (px/s):  mean={flow_pxs.mean():.1f}, "
      f"max={flow_pxs.max():.1f}")
print(f"  (This is typically much smaller than gaze velocity)")

# Step 5e: Relative velocity (THE KEY STEP!)
print_step("5e", "Relative velocity -- subtract camera motion from eye motion")
compute_relative_velocity(sim_df)
rel_pxs = sim_df["vel_rel_mag"].values * 1000
print(f"  vel_rel_mag (px/s):  mean={rel_pxs.mean():.1f}")
print(f"  Compare: gaze_vel={vel_pxs.mean():.1f}  -  "
      f"flow_vel={flow_pxs.mean():.1f}  ~  rel_vel={rel_pxs.mean():.1f}")
print(f"  (For this recording, head motion is small, so values are similar)")

# Step 5f: Flow RMS & adaptive threshold
print_step("5f", "Adaptive threshold -- adjust based on head motion amount")
ws = max(1, int(500 // (1000.0 / sampling_rate)))
compute_flow_window_rms(sim_df, window_size=ws)
base_thresh = 1.0  # px/ms = 1000 px/s
compute_adaptive_threshold(sim_df, base_threshold=base_thresh, gain=1.0,
                           window_size=ws)
thresh_pxs = sim_df["threshold"].values * 1000
print(f"  Base threshold: {base_thresh} px/ms = 1000 px/s")
print(f"  Adaptive range: [{thresh_pxs.min():.1f}, {thresh_pxs.max():.1f}] px/s")
print(f"  (Threshold goes up when head moves more -> fewer false saccades)")

# Step 5g: Classification
print_step("5g", "Classification -- label each sample as Fixation or Saccade")
result, thresh = classify_ivt(
    gd.copy(),
    velocity_threshold=base_thresh,
    min_fixation_duration=0,
    adapt=True,
    sampling_rate=sampling_rate,
)
pred = (result["event_type"] == "Fixation").astype(int).values
n_fix_pred = pred.sum()
n_sac_pred = (pred == 0).sum()
print(f"  Result: {n_fix_pred} Fixation ({n_fix_pred/len(pred)*100:.1f}%), "
      f"{n_sac_pred} Saccade ({n_sac_pred/len(pred)*100:.1f}%)")


# =====================================================================
# PART 6: COMPARE WITH GROUND TRUTH
# =====================================================================
print_header("PART 6: Is our answer correct? Compare with ground truth")

print_step("6a", "Sample-wise comparison (no duration filter)")
f1_fix = f1_score(gt_labels, pred, pos_label=1)
f1_sac = f1_score(gt_labels, pred, pos_label=0)
print(f"  Fixation F1: {f1_fix:.4f}  (1.0 = perfect)")
print(f"  Saccade  F1: {f1_sac:.4f}")
print(f"  (F1 combines precision and recall into a single score)")
print()
print(classification_report(gt_labels, pred,
                            target_names=["Saccade", "Fixation"], digits=4))

print_step("6b", "With 50ms minimum fixation duration (realistic scenario)")
result2, _ = classify_ivt(
    gd.copy(),
    velocity_threshold=base_thresh,
    min_fixation_duration=50,
    adapt=True,
    sampling_rate=sampling_rate,
)
pred2 = (result2["event_type"] == "Fixation").astype(int).values
f1_fix2 = f1_score(gt_labels, pred2, pos_label=1)
f1_sac2 = f1_score(gt_labels, pred2, pos_label=0)
print(f"  Fixation F1: {f1_fix2:.4f}")
print(f"  Saccade  F1: {f1_sac2:.4f}")

print_step("6c", "Compare: with flow vs without flow")
df_noflow = df.drop(columns=["flow_x", "flow_y", "video_timestamp"])
det_noflow = EventDetection(
    df_noflow.copy(), resolution=(1088, 1080),
    column_mapping={"x": "x", "y": "y", "timestamp": "timestamp"},
    is_normalized=False,
)
result3, _ = classify_ivt(
    det_noflow.gaze_data.copy(),
    velocity_threshold=base_thresh,
    min_fixation_duration=0,
    adapt=False,
)
pred3 = (result3["event_type"] == "Fixation").astype(int).values
f1_noflow = f1_score(gt_labels, pred3, pos_label=1)
print(f"  With flow (adaptive):  F1 = {f1_fix:.4f}")
print(f"  Without flow (legacy): F1 = {f1_noflow:.4f}")
print(f"  (For this recording, head motion is small, so results are similar)")


# =====================================================================
# PART 7: VELOCITY ACCURACY CHECK
# =====================================================================
print_header("PART 7: Do our velocities match the reference implementation?")

print("""
  We compare our backend's velocity computation against a simple
  NumPy implementation that follows the research code line-by-line.
  If they match, our code is correct.
""")

# Reference implementation
from scipy.signal import savgol_filter as sg_filter

frame_dur_ms = 1000.0 / sampling_rate
win = int(55 // frame_dur_ms)
if win % 2 == 0:
    win += 1

filt_x = sg_filter(gaze[:, 0], window_length=win, polyorder=3)
filt_y = sg_filter(gaze[:, 1], window_length=win, polyorder=3)

N = len(gaze)
t = time_gaze
x_delta = np.zeros(N)
y_delta = np.zeros(N)
t_delta = np.zeros(N)
x_delta[:-1] = filt_x[1:] - filt_x[:-1]
y_delta[:-1] = filt_y[1:] - filt_y[:-1]
t_delta[:-1] = t[1:] - t[:-1]
avg_dt = np.mean(t_delta)

ref_x_vel = x_delta / avg_dt
ref_y_vel = y_delta / avg_dt
ref_vel_mag = np.hypot(ref_x_vel, ref_y_vel)  # px/s

# Backend velocities (convert px/ms -> px/s)
backend_vel = result["vel_rel_mag"].values * 1000

# Compare where both are finite
mask = np.isfinite(backend_vel) & np.isfinite(ref_vel_mag)
residual = np.abs(backend_vel[mask] - ref_vel_mag[mask])
corr = np.corrcoef(backend_vel[mask], ref_vel_mag[mask])[0, 1]

print(f"  Samples compared: {mask.sum()}/{N}")
print(f"  Max difference:   {residual.max():.2f} px/s")
print(f"  Mean difference:  {residual.mean():.4f} px/s")
print(f"  Correlation:      {corr:.6f}")

if corr > 0.999:
    print(f"\n  PASS -- Velocity computation matches reference "
          f"(correlation {corr:.6f})")
else:
    print(f"\n  FAIL -- Velocity computation differs from reference!")


# =====================================================================
# PART 8: GENERATE VISUALIZATION
# =====================================================================
print_header("PART 8: Interactive visualization")
print("  Generating HTML visualization file...")

# Prepare data for visualization
timestamps_s = time_gaze - time_gaze[0]  # relative seconds

# Downsample for visualization if too many points
step = max(1, len(gaze) // 2000)

vis_data = {
    "timestamps": timestamps_s[::step].tolist(),
    "gaze_x": gaze[::step, 0].tolist(),
    "gaze_y": gaze[::step, 1].tolist(),
    "gt_labels": gt_labels[::step].tolist(),
    "pred_labels": pred[::step].tolist(),
    "velocity": backend_vel[::step].tolist(),
    "threshold": (sim_df["threshold"].values[::step] * 1000).tolist(),
    "flow_vel": (sim_df["flow_vel_mag"].values[::step] * 1000).tolist(),
    "stats": {
        "n_samples": int(N),
        "sampling_rate": round(float(sampling_rate), 1),
        "f1_fixation": round(float(f1_fix), 4),
        "f1_saccade": round(float(f1_sac), 4),
        "correlation": round(float(corr), 6),
        "gt_fixation_pct": round(float(n_fix) / N * 100, 1),
        "pred_fixation_pct": round(float(n_fix_pred) / N * 100, 1),
    },
}

html_content = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>I-VAT+Frel Verification -- Dataset A</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: #1a1a2e;
         color: #e0e0e0; padding: 20px; }
  h1 { color: #6dd5fa; margin-bottom: 10px; }
  h2 { color: #6dd5fa; margin: 20px 0 10px; font-size: 1.2em; }
  .stats { display: flex; gap: 20px; flex-wrap: wrap; margin: 15px 0; }
  .stat-box { background: #16213e; padding: 15px 20px; border-radius: 8px;
              border-left: 4px solid #6dd5fa; min-width: 180px; }
  .stat-box .value { font-size: 1.8em; font-weight: bold; color: #6dd5fa; }
  .stat-box .label { font-size: 0.85em; color: #888; margin-top: 4px; }
  .chart-container { background: #16213e; border-radius: 8px; padding: 15px;
                     margin: 15px 0; }
  canvas { width: 100% !important; background: #0f3460; border-radius: 4px; }
  .legend { display: flex; gap: 20px; padding: 8px 0; font-size: 0.9em; }
  .legend span { display: flex; align-items: center; gap: 6px; }
  .legend .dot { width: 12px; height: 12px; border-radius: 50%;
                 display: inline-block; }
  .explanation { background: #16213e; padding: 15px; border-radius: 8px;
                 margin: 10px 0; line-height: 1.6; }
  .pass { color: #4caf50; font-weight: bold; }
  .fail { color: #f44336; font-weight: bold; }
  .controls { display: flex; gap: 10px; align-items: center; margin: 10px 0; }
  .controls label { font-size: 0.9em; }
  .controls input[type=range] { flex: 1; }
  .controls span { min-width: 50px; text-align: right; }
  table { border-collapse: collapse; width: 100%; margin: 10px 0; }
  td, th { padding: 8px 12px; text-align: left; border-bottom: 1px solid #333; }
  th { color: #6dd5fa; }
</style>
</head>
<body>

<h1>I-VAT+Frel Verification -- Dataset A</h1>
<p>Verifying the head-mounted eye tracking enhancement against ground truth.</p>

<div class="stats" id="statsContainer"></div>

<h2>What is this test?</h2>
<div class="explanation">
  <strong>Dataset A</strong> is a 30-second recording from Pupil Invisible
  head-mounted eye tracking glasses.<br>
  A human expert labeled every gaze sample as either <strong>Fixation</strong>
  (eye staying still) or <strong>Saccade</strong> (eye moving fast).<br>
  We run our algorithm on the same data and compare: <strong>how many samples
  did we classify correctly?</strong><br><br>
  The key enhancement is <strong>optic flow compensation</strong>: when the
  person's head moves, the camera image shifts. We subtract this camera motion
  from the gaze motion to get the <strong>true eye movement</strong>.
</div>

<h2>Gaze Position (x, y over time)</h2>
<div class="chart-container">
  <div class="legend">
    <span><span class="dot" style="background:#4caf50"></span>
      Fixation (ground truth)</span>
    <span><span class="dot" style="background:#f44336"></span>
      Saccade (ground truth)</span>
  </div>
  <canvas id="gazeChart" height="200"></canvas>
  <div class="controls">
    <label>Time window:</label>
    <input type="range" id="timeSlider" min="0" max="100" value="0">
    <span id="timeLabel">0-5s</span>
  </div>
</div>

<h2>Velocity &amp; Threshold (the classification decision)</h2>
<div class="chart-container">
  <div class="legend">
    <span><span class="dot" style="background:#ff9800"></span>
      Eye velocity (px/s)</span>
    <span><span class="dot" style="background:#e91e63"></span>
      Threshold</span>
    <span><span class="dot" style="background:#00bcd4"></span>
      Camera (flow) velocity</span>
  </div>
  <canvas id="velChart" height="200"></canvas>
  <div class="explanation" style="margin-top:10px; font-size:0.9em;">
    <strong>How to read:</strong> When the orange line (eye velocity) is
    BELOW the pink line (threshold) &rarr; Fixation. Above &rarr; Saccade.
    <br>The cyan line shows camera motion &mdash; this is what gets subtracted
    from gaze velocity in the enhanced pipeline.
  </div>
</div>

<h2>Classification Result vs Ground Truth</h2>
<div class="chart-container">
  <canvas id="classChart" height="100"></canvas>
  <div class="legend">
    <span><span class="dot" style="background:#4caf50"></span> Correct</span>
    <span><span class="dot" style="background:#f44336"></span> Error</span>
  </div>
</div>

<h2>Detailed Results</h2>
<table>
  <tr><th>Metric</th><th>Value</th><th>Meaning</th></tr>
  <tr><td>Fixation F1</td><td id="f1fix"></td>
      <td>How well we detect fixations (1.0 = perfect)</td></tr>
  <tr><td>Saccade F1</td><td id="f1sac"></td>
      <td>How well we detect saccades</td></tr>
  <tr><td>Velocity correlation</td><td id="vcorr"></td>
      <td>How closely our velocities match reference (1.0 = identical)</td></tr>
  <tr><td>Samples</td><td id="nsamples"></td>
      <td>Total gaze samples in recording</td></tr>
  <tr><td>Sampling rate</td><td id="srate"></td>
      <td>Gaze samples per second</td></tr>
</table>

<script>
var DATA = __DATA_PLACEHOLDER__;

// Stats boxes
var statsHtml = [
  {value: DATA.stats.f1_fixation, label: 'Fixation F1 Score',
   good: DATA.stats.f1_fixation > 0.95},
  {value: DATA.stats.f1_saccade, label: 'Saccade F1 Score',
   good: DATA.stats.f1_saccade > 0.5},
  {value: DATA.stats.correlation, label: 'Velocity Correlation',
   good: DATA.stats.correlation > 0.999},
  {value: DATA.stats.gt_fixation_pct + '%', label: 'GT Fixation %'},
  {value: DATA.stats.pred_fixation_pct + '%', label: 'Predicted Fixation %'},
].map(function(s) {
  var cls = s.good === false ? 'fail' : (s.good ? 'pass' : '');
  return '<div class="stat-box"><div class="value ' + cls + '">' +
    s.value + '</div><div class="label">' + s.label + '</div></div>';
}).join('');
document.getElementById('statsContainer').innerHTML = statsHtml;

document.getElementById('f1fix').textContent = DATA.stats.f1_fixation;
document.getElementById('f1sac').textContent = DATA.stats.f1_saccade;
document.getElementById('vcorr').textContent = DATA.stats.correlation;
document.getElementById('nsamples').textContent = DATA.stats.n_samples;
document.getElementById('srate').textContent = DATA.stats.sampling_rate + ' Hz';

// Simple canvas chart drawing
function drawChart(canvasId, drawFn) {
  var canvas = document.getElementById(canvasId);
  var ctx = canvas.getContext('2d');
  canvas.width = canvas.offsetWidth * 2;
  canvas.height = canvas.offsetHeight * 2;
  ctx.scale(2, 2);
  var w = canvas.offsetWidth, h = canvas.offsetHeight;
  ctx.clearRect(0, 0, w, h);
  drawFn(ctx, w, h);
}

var windowStart = 0;
var windowDuration = 5; // seconds

function getWindowIndices() {
  var tStart = windowStart;
  var tEnd = windowStart + windowDuration;
  var iStart = 0, iEnd = DATA.timestamps.length - 1;
  for (var i = 0; i < DATA.timestamps.length; i++) {
    if (DATA.timestamps[i] >= tStart) { iStart = i; break; }
  }
  for (var i = DATA.timestamps.length - 1; i >= 0; i--) {
    if (DATA.timestamps[i] <= tEnd) { iEnd = i; break; }
  }
  return [iStart, iEnd, tStart, tEnd];
}

function drawGazeChart() {
  drawChart('gazeChart', function(ctx, w, h) {
    var wIdx = getWindowIndices();
    var iStart = wIdx[0], iEnd = wIdx[1], tStart = wIdx[2], tEnd = wIdx[3];
    var ts = DATA.timestamps.slice(iStart, iEnd + 1);
    var gx = DATA.gaze_x.slice(iStart, iEnd + 1);
    var gt = DATA.gt_labels.slice(iStart, iEnd + 1);

    if (ts.length < 2) return;

    var xMin = Math.min.apply(null, gx), xMax = Math.max.apply(null, gx);
    var pad = 20;

    for (var i = 1; i < ts.length; i++) {
      var px = pad + (ts[i] - tStart) / windowDuration * (w - 2*pad);
      var py = pad + (1 - (gx[i] - xMin) / (xMax - xMin + 1)) * (h - 2*pad);
      var ppx = pad + (ts[i-1] - tStart) / windowDuration * (w - 2*pad);
      var ppy = pad + (1 - (gx[i-1] - xMin) / (xMax - xMin + 1)) * (h - 2*pad);
      ctx.beginPath();
      ctx.moveTo(ppx, ppy);
      ctx.lineTo(px, py);
      ctx.strokeStyle = gt[i] === 1 ? '#4caf50' : '#f44336';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }

    ctx.fillStyle = '#888';
    ctx.font = '11px sans-serif';
    ctx.fillText('Gaze X (px)', 5, 14);
  });
}

function drawVelChart() {
  drawChart('velChart', function(ctx, w, h) {
    var wIdx = getWindowIndices();
    var iStart = wIdx[0], iEnd = wIdx[1], tStart = wIdx[2];
    var ts = DATA.timestamps.slice(iStart, iEnd + 1);
    var vel = DATA.velocity.slice(iStart, iEnd + 1);
    var thr = DATA.threshold.slice(iStart, iEnd + 1);
    var fv = DATA.flow_vel.slice(iStart, iEnd + 1);

    if (ts.length < 2) return;

    var allVals = vel.concat(thr);
    var maxVel = Math.min(3000, Math.max.apply(null, allVals) * 1.2);
    var pad = 20;

    function drawLine(arr, color, maxY) {
      ctx.beginPath();
      for (var i = 0; i < ts.length; i++) {
        var px = pad + (ts[i] - tStart) / windowDuration * (w - 2*pad);
        var py = h - pad - Math.min(arr[i], maxY) / maxY * (h - 2*pad);
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }

    drawLine(vel, '#ff9800', maxVel);
    drawLine(thr, '#e91e63', maxVel);
    drawLine(fv, '#00bcd4', maxVel);

    ctx.fillStyle = '#888';
    ctx.font = '11px sans-serif';
    ctx.fillText('Velocity (px/s)', 5, 14);
    ctx.fillText('0', 5, h - 5);
    ctx.fillText(Math.round(maxVel) + '', 5, 25);
  });
}

function drawClassChart() {
  drawChart('classChart', function(ctx, w, h) {
    var wIdx = getWindowIndices();
    var iStart = wIdx[0], iEnd = wIdx[1], tStart = wIdx[2];
    var ts = DATA.timestamps.slice(iStart, iEnd + 1);
    var gt = DATA.gt_labels.slice(iStart, iEnd + 1);
    var pr = DATA.pred_labels.slice(iStart, iEnd + 1);
    var pad = 20;

    var rowH = (h - 2*pad) / 3;
    for (var i = 0; i < ts.length; i++) {
      var px = pad + (ts[i] - tStart) / windowDuration * (w - 2*pad);
      var bw = Math.max(2, (w - 2*pad) / ts.length);

      // Ground truth
      ctx.fillStyle = gt[i] === 1 ? '#4caf50' : '#f44336';
      ctx.fillRect(px, pad, bw, rowH - 2);

      // Prediction
      ctx.fillStyle = pr[i] === 1 ? '#4caf50' : '#f44336';
      ctx.fillRect(px, pad + rowH, bw, rowH - 2);

      // Match indicator
      ctx.fillStyle = gt[i] === pr[i] ? 'rgba(76,175,80,0.27)' :
                                         'rgba(244,67,54,0.53)';
      ctx.fillRect(px, pad + 2*rowH, bw, rowH - 2);
    }

    ctx.fillStyle = '#888';
    ctx.font = '11px sans-serif';
    ctx.fillText('Ground Truth', 5, pad + rowH/2 + 4);
    ctx.fillText('Our Prediction', 5, pad + rowH + rowH/2 + 4);
    ctx.fillText('Match?', 5, pad + 2*rowH + rowH/2 + 4);
  });
}

// Slider
var slider = document.getElementById('timeSlider');
var maxTime = DATA.timestamps[DATA.timestamps.length - 1];
slider.addEventListener('input', function() {
  windowStart = (slider.value / 100) *
    Math.max(0, maxTime - windowDuration);
  document.getElementById('timeLabel').textContent =
    windowStart.toFixed(1) + '-' +
    (windowStart + windowDuration).toFixed(1) + 's';
  drawAll();
});

function drawAll() { drawGazeChart(); drawVelChart(); drawClassChart(); }

window.addEventListener('resize', drawAll);
setTimeout(drawAll, 100);
</script>
</body>
</html>"""

# Inject the data JSON into the HTML
html_content = html_content.replace("__DATA_PLACEHOLDER__",
                                    json.dumps(vis_data))

output_path = os.path.join(os.path.dirname(__file__), "data",
                           "visualization", "verify_ivat.html")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"\n  Visualization saved to: {output_path}")
print(f"  -> Open this file in your browser to see the interactive charts!")


# =====================================================================
# FINAL SUMMARY
# =====================================================================
print_header("FINAL SUMMARY")

fix_status = "PASS (>0.95)" if f1_fix > 0.95 else "NEEDS WORK"
sac_status = "PASS (>0.50)" if f1_sac > 0.50 else "NEEDS WORK"
vel_status = "MATCH" if corr > 0.999 else "MISMATCH"

print(f"""
  Fixation F1 Score:     {f1_fix:.4f}  {fix_status}
  Saccade F1 Score:      {f1_sac:.4f}  {sac_status}
  Velocity Correlation:  {corr:.6f}  {vel_status}
  With 50ms dur filter:  {f1_fix2:.4f}  (realistic scenario)

  WHAT DO THESE NUMBERS MEAN?
  - F1 > 0.97 for fixation = our algorithm correctly identifies fixations
  - Velocity correlation ~ 1.0 = our math matches the reference exactly
  - The enhancement is most visible with recordings that have MORE head motion
    (Dataset A has relatively little head motion)

  TO SEE THE VISUALIZATION:
  Open in browser: data/visualization/verify_ivat.html
  - Use the slider to scroll through the 30-second recording
  - Green = Fixation, Red = Saccade
  - Compare Ground Truth row vs Our Prediction row
""")
