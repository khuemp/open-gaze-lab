"""
=============================================================================
 MULTI-DATASET VERIFICATION for I-VAT+Frel (Head-Mounted Eye Tracking)
=============================================================================

Runs the same verification as verify_ivat.py across ALL 8 datasets:
  - dataset/dynamic/A, B, C, D  (subject moving head)
  - dataset/static/A, B, C, D   (subject mostly stationary)

For each dataset it checks:
  1. With-flow (adaptive) classification  -> F1 scores
  2. With-flow + 50ms duration filter     -> F1 scores
  3. Without-flow (legacy) classification -> F1 scores
  4. Velocity accuracy vs reference impl  -> Pearson correlation

HOW TO RUN:
  cd event_detection-backend
  python verify_all_datasets.py
=============================================================================
"""

import sys
import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter as sg_filter
from sklearn.metrics import f1_score
import json

sys.path.insert(0, os.path.dirname(__file__))
from src import EventDetection
from src.detection_algorithms import classify_ivt
from src.preprocessing import (
    apply_savgol_filter, compute_gaze_velocity, compute_flow_velocity,
    compute_relative_velocity, compute_flow_window_rms,
    compute_adaptive_threshold,
)

# ── Configuration ─────────────────────────────────────────────────────
RESOLUTION = (1088, 1080)          # Pupil Invisible scene camera
VELOCITY_THRESHOLD = 1.0           # px/ms  (= 1000 px/s)
SAVGOL_WINDOW_MS = 55.0
GAIN = 1.0
COLUMN_MAPPING_FLOW = {
    "x": "x", "y": "y", "timestamp": "timestamp",
    "flow_x": "flow_x", "flow_y": "flow_y",
    "video_timestamp": "video_timestamp",
}
COLUMN_MAPPING_NO_FLOW = {
    "x": "x", "y": "y", "timestamp": "timestamp",
}

DATASET_ROOT = os.path.join(os.path.dirname(__file__), "dataset")
CATEGORIES = ["dynamic", "static"]
SUBJECTS = ["A", "B", "C", "D"]


# ── Helpers ───────────────────────────────────────────────────────────
def load_dataset(data_dir):
    """Load .npy files from a dataset folder and return raw arrays."""
    gaze = np.load(os.path.join(data_dir, "gaze.npy"))
    time_gaze = np.load(os.path.join(data_dir, "time_gaze.npy"))
    gt_labels = np.load(os.path.join(data_dir, "gt_labels.npy"))
    optic_flow = np.load(os.path.join(data_dir, "optic_flow.npy"))
    time_of = np.load(os.path.join(data_dir, "time_optic_flow.npy"))
    return gaze, time_gaze, gt_labels, optic_flow, time_of


def prepare_dataframe(gaze, time_gaze, optic_flow, time_of):
    """Average optic flow, map to gaze samples, build DataFrame."""
    mean_flow = np.nanmean(optic_flow.astype(np.float64), axis=(1, 2))
    indices = np.searchsorted(time_of, time_gaze, side="right") - 1
    indices = np.clip(indices, 0, len(time_of) - 1)
    flow_per_gaze = mean_flow[indices]
    video_ts_per_gaze = time_of[indices]

    df = pd.DataFrame({
        "timestamp": time_gaze,
        "x": gaze[:, 0],
        "y": gaze[:, 1],
        "flow_x": flow_per_gaze[:, 0],
        "flow_y": flow_per_gaze[:, 1],
        "video_timestamp": video_ts_per_gaze,
    })
    return df


def compute_reference_velocity(gaze, time_gaze, sampling_rate):
    """Reference velocity implementation (NumPy, matches research code)."""
    frame_dur_ms = 1000.0 / sampling_rate
    win = int(55 // frame_dur_ms)
    if win % 2 == 0:
        win += 1

    filt_x = sg_filter(gaze[:, 0], window_length=win, polyorder=3)
    filt_y = sg_filter(gaze[:, 1], window_length=win, polyorder=3)

    N = len(gaze)
    x_delta = np.zeros(N)
    y_delta = np.zeros(N)
    t_delta = np.zeros(N)
    x_delta[:-1] = filt_x[1:] - filt_x[:-1]
    y_delta[:-1] = filt_y[1:] - filt_y[:-1]
    t_delta[:-1] = time_gaze[1:] - time_gaze[:-1]
    avg_dt = np.mean(t_delta)

    ref_x_vel = x_delta / avg_dt
    ref_y_vel = y_delta / avg_dt
    ref_vel_mag = np.hypot(ref_x_vel, ref_y_vel)  # px/s
    return ref_vel_mag


def verify_dataset(category, subject):
    """Run full verification pipeline on one dataset. Returns result dict."""
    data_dir = os.path.join(DATASET_ROOT, category, subject)
    label = f"{category}/{subject}"

    gaze, time_gaze, gt_labels, optic_flow, time_of = load_dataset(data_dir)
    N = len(gaze)
    sampling_rate = 1.0 / np.median(np.diff(time_gaze))
    duration_s = time_gaze[-1] - time_gaze[0]

    # Sanity check: gaze within resolution bounds
    gaze_max_x, gaze_max_y = np.nanmax(gaze[:, 0]), np.nanmax(gaze[:, 1])
    if gaze_max_x > RESOLUTION[0] or gaze_max_y > RESOLUTION[1]:
        print(f"  WARNING [{label}]: gaze max ({gaze_max_x:.0f}, {gaze_max_y:.0f}) "
              f"exceeds resolution {RESOLUTION}")

    df = prepare_dataframe(gaze, time_gaze, optic_flow, time_of)

    # ── Test 1: With flow, adaptive, no duration filter ───────────
    det_flow = EventDetection(
        df.copy(), resolution=RESOLUTION,
        column_mapping=COLUMN_MAPPING_FLOW, is_normalized=False,
    )
    result_flow, _ = classify_ivt(
        det_flow.gaze_data.copy(),
        velocity_threshold=VELOCITY_THRESHOLD,
        min_fixation_duration=0,
        adapt=True,
        sampling_rate=sampling_rate,
    )
    pred_flow = (result_flow["event_type"] == "Fixation").astype(int).values
    f1_fix_flow = f1_score(gt_labels, pred_flow, pos_label=1)
    f1_sac_flow = f1_score(gt_labels, pred_flow, pos_label=0)

    # ── Test 2: With flow, adaptive, 50ms duration filter ─────────
    result_flow50, _ = classify_ivt(
        det_flow.gaze_data.copy(),
        velocity_threshold=VELOCITY_THRESHOLD,
        min_fixation_duration=50,
        adapt=True,
        sampling_rate=sampling_rate,
    )
    pred_flow50 = (result_flow50["event_type"] == "Fixation").astype(int).values
    f1_fix_flow50 = f1_score(gt_labels, pred_flow50, pos_label=1)
    f1_sac_flow50 = f1_score(gt_labels, pred_flow50, pos_label=0)

    # ── Test 3: Without flow (legacy I-VT) ────────────────────────
    df_noflow = df.drop(columns=["flow_x", "flow_y", "video_timestamp"])
    det_noflow = EventDetection(
        df_noflow.copy(), resolution=RESOLUTION,
        column_mapping=COLUMN_MAPPING_NO_FLOW, is_normalized=False,
    )
    result_noflow, _ = classify_ivt(
        det_noflow.gaze_data.copy(),
        velocity_threshold=VELOCITY_THRESHOLD,
        min_fixation_duration=0,
        adapt=False,
    )
    pred_noflow = (result_noflow["event_type"] == "Fixation").astype(int).values
    f1_fix_noflow = f1_score(gt_labels, pred_noflow, pos_label=1)
    f1_sac_noflow = f1_score(gt_labels, pred_noflow, pos_label=0)

    # ── Test 4: Velocity correlation vs reference ─────────────────
    # Compare gaze velocity (vel_mag, before flow subtraction) against the
    # NumPy reference, since the reference doesn't do flow compensation.
    ref_vel = compute_reference_velocity(gaze, time_gaze, sampling_rate)
    backend_vel = result_flow["vel_mag"].values * 1000  # px/ms -> px/s
    mask = np.isfinite(backend_vel) & np.isfinite(ref_vel)
    if mask.sum() > 2:
        corr = np.corrcoef(backend_vel[mask], ref_vel[mask])[0, 1]
        residual_mean = np.abs(backend_vel[mask] - ref_vel[mask]).mean()
    else:
        corr = float("nan")
        residual_mean = float("nan")

    # ── Collect visualization data ────────────────────────────────
    sim_df = det_flow.gaze_data.copy()
    apply_savgol_filter(sim_df, sampling_rate, window_size_ms=SAVGOL_WINDOW_MS)
    compute_gaze_velocity(sim_df, x_col="filter_x", y_col="filter_y")
    compute_flow_velocity(sim_df)
    compute_relative_velocity(sim_df)
    ws = max(1, int(500 // (1000.0 / sampling_rate)))
    compute_flow_window_rms(sim_df, window_size=ws)
    compute_adaptive_threshold(sim_df, base_threshold=VELOCITY_THRESHOLD,
                               gain=GAIN, window_size=ws)

    timestamps_s = time_gaze - time_gaze[0]
    step = max(1, N // 2000)

    vis_data = {
        "label": f"{category}/{subject}",
        "category": category,
        "subject": subject,
        "timestamps": timestamps_s[::step].tolist(),
        "gaze_x": gaze[::step, 0].tolist(),
        "gaze_y": gaze[::step, 1].tolist(),
        "gt_labels": gt_labels[::step].tolist(),
        "pred_labels": pred_flow[::step].tolist(),
        "pred_noflow_labels": pred_noflow[::step].tolist(),
        "velocity": np.nan_to_num(sim_df["vel_rel_mag"].values[::step] * 1000).tolist(),
        "threshold": np.nan_to_num(sim_df["threshold"].values[::step] * 1000).tolist(),
        "flow_vel": np.nan_to_num(sim_df["flow_vel_mag"].values[::step] * 1000).tolist(),
    }

    metrics = {
        "category": category,
        "subject": subject,
        "n_samples": int(N),
        "sampling_rate": round(float(sampling_rate), 1),
        "duration_s": round(float(duration_s), 1),
        "has_nan_gaze": bool(np.isnan(gaze).any()),
        "f1_fix_flow": round(float(f1_fix_flow), 4),
        "f1_sac_flow": round(float(f1_sac_flow), 4),
        "f1_fix_flow50": round(float(f1_fix_flow50), 4),
        "f1_sac_flow50": round(float(f1_sac_flow50), 4),
        "f1_fix_noflow": round(float(f1_fix_noflow), 4),
        "f1_sac_noflow": round(float(f1_sac_noflow), 4),
        "f1_improvement": round(float(f1_fix_flow - f1_fix_noflow), 4),
        "vel_correlation": round(float(corr), 6),
        "vel_residual_mean": round(float(residual_mean), 4),
        "vel_pass": bool(corr > 0.99),
    }

    return metrics, vis_data


# ── HTML Report Generation ────────────────────────────────────────────
def generate_html(ok_results, vis_data_list):
    """Generate interactive HTML report with per-dataset charts."""

    html_template = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>I-VAT+Frel Multi-Dataset Verification</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: #1a1a2e;
         color: #e0e0e0; padding: 20px; }
  h1 { color: #6dd5fa; margin-bottom: 5px; }
  h2 { color: #6dd5fa; margin: 18px 0 8px; font-size: 1.15em; }
  .sub { color: #888; margin-bottom: 15px; }
  .verdict { text-align: center; padding: 14px; border-radius: 8px;
             font-size: 1.4em; font-weight: bold; margin: 15px 0; }
  .verdict.pass { background: #1b5e2033; color: #4caf50;
                  border: 2px solid #4caf50; }
  .verdict.fail { background: #b71c1c33; color: #f44336;
                  border: 2px solid #f44336; }
  .stats { display: flex; gap: 12px; flex-wrap: wrap; margin: 12px 0; }
  .sb { background: #16213e; padding: 12px 16px; border-radius: 8px;
        border-left: 4px solid #6dd5fa; min-width: 150px; flex: 1; }
  .sb .v { font-size: 1.5em; font-weight: bold; color: #6dd5fa; }
  .sb .l { font-size: .8em; color: #888; margin-top: 3px; }
  .sb.g .v { color: #4caf50; }
  .sb.b .v { color: #f44336; }
  table { border-collapse: collapse; width: 100%; margin: 10px 0; }
  td, th { padding: 7px 10px; text-align: left;
           border-bottom: 1px solid #333; }
  th { color: #6dd5fa; }
  tr.ck { cursor: pointer; }
  tr.ck:hover { background: #16213e; }
  tr.ar { background: #16213e88; }
  .tabs { display: flex; gap: 4px; flex-wrap: wrap; margin: 15px 0 0; }
  .tb { padding: 7px 14px; border: 2px solid #333; border-bottom: none;
        border-radius: 8px 8px 0 0; background: #16213e; color: #888;
        cursor: pointer; font-size: .88em; }
  .tb:hover { color: #e0e0e0; border-color: #555; }
  .tb.a { color: #e0e0e0; background: #0f3460; }
  .tb.dy { border-top-color: #ff9800; }
  .tb.st { border-top-color: #2196f3; }
  .tb.a.dy { border-color: #ff9800; }
  .tb.a.st { border-color: #2196f3; }
  .tc { background: #0f3460; border: 2px solid #6dd5fa;
        border-radius: 0 8px 8px 8px; padding: 20px; }
  .cc { background: #16213e; border-radius: 8px; padding: 12px;
        margin: 12px 0; }
  canvas { width: 100% !important; background: #0f3460;
           border-radius: 4px; }
  .leg { display: flex; gap: 16px; padding: 6px 0; font-size: .88em;
         flex-wrap: wrap; }
  .leg span { display: flex; align-items: center; gap: 5px; }
  .leg .d { width: 11px; height: 11px; border-radius: 50%;
            display: inline-block; }
  .ctl { display: flex; gap: 10px; align-items: center; margin: 10px 0; }
  .ctl label { font-size: .88em; }
  .ctl input[type=range] { flex: 1; }
  .ctl span { min-width: 80px; text-align: right; }
  .pt { color: #4caf50; }
  .ft { color: #f44336; }
  .info { background: #16213e; padding: 12px; border-radius: 8px;
          margin: 8px 0; line-height: 1.5; font-size: .9em; }
</style>
</head>
<body>

<h1>I-VAT+Frel Multi-Dataset Verification</h1>
<p class="sub">Verifying head-mounted eye tracking enhancement across 8
  datasets (dynamic/static &times; A-D)</p>

<div class="verdict" id="vb"></div>
<div class="stats" id="as"></div>

<h2>Summary Table</h2>
<p class="sub">Click a row to jump to that dataset's charts</p>
<table id="stbl">
  <thead><tr>
    <th>Dataset</th><th>Samples</th><th>Hz</th>
    <th>F1-Fix (flow)</th><th>F1-Sac (flow)</th>
    <th>F1-Fix (no flow)</th><th>Improvement</th>
    <th>Vel Corr</th><th>Status</th>
  </tr></thead>
  <tbody></tbody>
</table>

<h2>Dataset Details</h2>
<div class="tabs" id="tbs"></div>
<div class="tc" id="tc">
  <div class="stats" id="ds"></div>

  <h2>Gaze Position (x, y over time)</h2>
  <div class="cc">
    <div class="leg">
      <span><span class="d" style="background:#4caf50"></span>
        Fixation (GT)</span>
      <span><span class="d" style="background:#f44336"></span>
        Saccade (GT)</span>
    </div>
    <canvas id="cG" height="250"></canvas>
  </div>

  <h2>Velocity &amp; Threshold</h2>
  <div class="cc">
    <div class="leg">
      <span><span class="d" style="background:#ff9800"></span>
        Relative velocity (px/s)</span>
      <span><span class="d" style="background:#e91e63"></span>
        Adaptive threshold</span>
      <span><span class="d" style="background:#00bcd4"></span>
        Camera flow velocity</span>
    </div>
    <canvas id="cV" height="200"></canvas>
    <div class="info">
      <strong>How to read:</strong> When the orange line (eye velocity)
      is BELOW the pink line (threshold) &rarr; Fixation.
      Above &rarr; Saccade. Cyan shows camera motion subtracted in the
      enhanced pipeline.
    </div>
  </div>

  <h2>Classification Result vs Ground Truth</h2>
  <div class="cc">
    <canvas id="cC" height="120"></canvas>
    <div class="leg">
      <span><span class="d" style="background:#4caf50"></span>
        Correct</span>
      <span><span class="d" style="background:#f44336"></span>
        Error</span>
    </div>
  </div>

  <h2>With Flow vs Without Flow</h2>
  <div class="cc">
    <canvas id="cF" height="120"></canvas>
    <div class="leg">
      <span><span class="d" style="background:#4caf50"></span>
        Fixation</span>
      <span><span class="d" style="background:#f44336"></span>
        Saccade</span>
    </div>
  </div>

  <div class="ctl">
    <label>Time window:</label>
    <input type="range" id="sl" min="0" max="100" value="0">
    <span id="tl">0.0-5.0s</span>
  </div>

  <h2>Detailed Metrics</h2>
  <table id="dm"></table>
</div>

<script>
var D = __DATA_PLACEHOLDER__;
var R = __RESULTS_PLACEHOLDER__;
var ai = 0, ws = {}, wd = 5;

function avg(a) {
  return a.reduce(function(s, v) { return s + v; }, 0) / a.length;
}

function sb(v, l, g) {
  var c = g === true ? 'g' : (g === false ? 'b' : '');
  return '<div class="sb ' + c + '"><div class="v">' + v +
         '</div><div class="l">' + l + '</div></div>';
}

function pm(v) { return (v >= 0 ? '+' : '') + v; }

function init() {
  var ap = R.every(function(r) { return r.vel_pass; });
  var vb = document.getElementById('vb');
  vb.className = 'verdict ' + (ap ? 'pass' : 'fail');
  vb.textContent = ap
    ? 'OVERALL: PASS -- All ' + R.length + ' datasets verified'
    : 'OVERALL: ISSUES DETECTED';

  var mf = avg(R.map(function(r) { return r.f1_fix_flow; }));
  var mn = avg(R.map(function(r) { return r.f1_fix_noflow; }));
  var mi = avg(R.map(function(r) { return r.f1_improvement; }));
  var mc = avg(R.map(function(r) { return r.vel_correlation; }));

  document.getElementById('as').innerHTML = [
    sb(mf.toFixed(4), 'Mean F1 (flow)', mf > 0.8),
    sb(mn.toFixed(4), 'Mean F1 (no flow)', mn > 0.8),
    sb(pm(mi.toFixed(4)), 'Mean Improvement', mi > 0),
    sb(mc.toFixed(6), 'Mean Vel Correlation', mc > 0.99),
    sb(R.length, 'Datasets', null)
  ].join('');

  var tb = document.querySelector('#stbl tbody');
  R.forEach(function(r, i) {
    var tr = document.createElement('tr');
    tr.className = 'ck';
    tr.onclick = function() { sel(i); };
    var s = r.vel_pass
      ? '<span class="pt">PASS</span>'
      : '<span class="ft">FAIL</span>';
    tr.innerHTML =
      '<td>' + r.category + '/' + r.subject + '</td>' +
      '<td>' + r.n_samples + '</td><td>' + r.sampling_rate + '</td>' +
      '<td>' + r.f1_fix_flow + '</td>' +
      '<td>' + r.f1_sac_flow + '</td>' +
      '<td>' + r.f1_fix_noflow + '</td>' +
      '<td>' + pm(r.f1_improvement) + '</td>' +
      '<td>' + r.vel_correlation + '</td>' +
      '<td>' + s + '</td>';
    tb.appendChild(tr);
  });

  var tbs = document.getElementById('tbs');
  D.forEach(function(d, i) {
    var b = document.createElement('button');
    b.className = 'tb ' + (d.category === 'dynamic' ? 'dy' : 'st');
    b.textContent = d.label;
    b.onclick = function() { sel(i); };
    tbs.appendChild(b);
    ws[i] = 0;
  });
  sel(0);
}

function sel(idx) {
  ai = idx;
  document.querySelectorAll('.tb').forEach(function(b, i) {
    b.className = 'tb ' +
      (D[i].category === 'dynamic' ? 'dy' : 'st') +
      (i === idx ? ' a' : '');
  });
  document.querySelectorAll('#stbl tbody tr').forEach(function(r, i) {
    r.className = 'ck' + (i === idx ? ' ar' : '');
  });

  var tc = document.getElementById('tc');
  tc.style.borderColor =
    D[idx].category === 'dynamic' ? '#ff9800' : '#2196f3';

  var r = R[idx];
  document.getElementById('ds').innerHTML = [
    sb(r.f1_fix_flow, 'F1-Fix (flow)', r.f1_fix_flow > 0.9),
    sb(r.f1_sac_flow, 'F1-Sac (flow)', r.f1_sac_flow > 0.5),
    sb(r.f1_fix_noflow, 'F1-Fix (no flow)', null),
    sb(pm(r.f1_improvement), 'Flow Improvement', r.f1_improvement > 0),
    sb(r.vel_correlation, 'Vel Correlation', r.vel_pass),
    sb(r.duration_s + 's', 'Duration', null)
  ].join('');

  document.getElementById('dm').innerHTML =
    '<tr><th>Metric</th><th>Value</th></tr>' +
    '<tr><td>Samples</td><td>' + r.n_samples + '</td></tr>' +
    '<tr><td>Sampling rate</td><td>' + r.sampling_rate + ' Hz</td></tr>' +
    '<tr><td>F1-Fix (flow, adaptive)</td><td>' +
      r.f1_fix_flow + '</td></tr>' +
    '<tr><td>F1-Sac (flow, adaptive)</td><td>' +
      r.f1_sac_flow + '</td></tr>' +
    '<tr><td>F1-Fix (flow + 50ms)</td><td>' +
      r.f1_fix_flow50 + '</td></tr>' +
    '<tr><td>F1-Sac (flow + 50ms)</td><td>' +
      r.f1_sac_flow50 + '</td></tr>' +
    '<tr><td>F1-Fix (no flow)</td><td>' +
      r.f1_fix_noflow + '</td></tr>' +
    '<tr><td>F1-Sac (no flow)</td><td>' +
      r.f1_sac_noflow + '</td></tr>' +
    '<tr><td>Flow improvement</td><td>' +
      pm(r.f1_improvement) + '</td></tr>' +
    '<tr><td>Velocity correlation</td><td>' +
      r.vel_correlation + '</td></tr>' +
    '<tr><td>Velocity residual mean</td><td>' +
      r.vel_residual_mean + ' px/s</td></tr>';

  document.getElementById('sl').value = 0;
  ws[idx] = 0;
  utl();
  drawAll();
}

/* ── Chart drawing helpers ── */

function dc(id, fn) {
  var c = document.getElementById(id);
  if (!c) return;
  var x = c.getContext('2d');
  c.width = c.offsetWidth * 2;
  c.height = c.offsetHeight * 2;
  x.scale(2, 2);
  var w = c.offsetWidth, h = c.offsetHeight;
  x.clearRect(0, 0, w, h);
  fn(x, w, h);
}

function gwi(d) {
  var t0 = ws[ai] || 0, t1 = t0 + wd;
  var i0 = 0, i1 = d.timestamps.length - 1;
  for (var i = 0; i < d.timestamps.length; i++) {
    if (d.timestamps[i] >= t0) { i0 = i; break; }
  }
  for (var i = d.timestamps.length - 1; i >= 0; i--) {
    if (d.timestamps[i] <= t1) { i1 = i; break; }
  }
  return [i0, i1, t0, t1];
}

function drawGaze() {
  var d = D[ai];
  dc('cG', function(x, w, h) {
    var wi = gwi(d), i0 = wi[0], i1 = wi[1], t0 = wi[2];
    var ts = d.timestamps.slice(i0, i1 + 1);
    var gx = d.gaze_x.slice(i0, i1 + 1);
    var gy = d.gaze_y.slice(i0, i1 + 1);
    var gt = d.gt_labels.slice(i0, i1 + 1);
    if (ts.length < 2) return;

    var p = 20, hh = h / 2;

    /* X in top half */
    var xn = Math.min.apply(null, gx);
    var xx = Math.max.apply(null, gx);
    for (var i = 1; i < ts.length; i++) {
      var px = p + (ts[i] - t0) / wd * (w - 2*p);
      var py = p + (1 - (gx[i] - xn) / (xx - xn + 1)) * (hh - 2*p);
      var qx = p + (ts[i-1] - t0) / wd * (w - 2*p);
      var qy = p + (1 - (gx[i-1] - xn) / (xx - xn + 1)) * (hh - 2*p);
      x.beginPath(); x.moveTo(qx, qy); x.lineTo(px, py);
      x.strokeStyle = gt[i] === 1 ? '#4caf50' : '#f44336';
      x.lineWidth = 1.5; x.stroke();
    }

    /* Y in bottom half */
    var yn = Math.min.apply(null, gy);
    var yx = Math.max.apply(null, gy);
    for (var i = 1; i < ts.length; i++) {
      var px = p + (ts[i] - t0) / wd * (w - 2*p);
      var py = hh + p + (1 - (gy[i] - yn) / (yx - yn + 1)) * (hh - 2*p);
      var qx = p + (ts[i-1] - t0) / wd * (w - 2*p);
      var qy = hh + p + (1 - (gy[i-1] - yn) / (yx - yn + 1)) * (hh - 2*p);
      x.beginPath(); x.moveTo(qx, qy); x.lineTo(px, py);
      x.strokeStyle = gt[i] === 1 ? '#66bb6a' : '#ef5350';
      x.lineWidth = 1.5; x.stroke();
    }

    /* divider */
    x.beginPath(); x.moveTo(0, hh); x.lineTo(w, hh);
    x.strokeStyle = '#555'; x.lineWidth = 1; x.stroke();
    x.fillStyle = '#888'; x.font = '11px sans-serif';
    x.fillText('Gaze X (px)', 5, 14);
    x.fillText('Gaze Y (px)', 5, hh + 14);
  });
}

function drawVel() {
  var d = D[ai];
  dc('cV', function(x, w, h) {
    var wi = gwi(d), i0 = wi[0], i1 = wi[1], t0 = wi[2];
    var ts = d.timestamps.slice(i0, i1 + 1);
    var vel = d.velocity.slice(i0, i1 + 1);
    var thr = d.threshold.slice(i0, i1 + 1);
    var fv = d.flow_vel.slice(i0, i1 + 1);
    if (ts.length < 2) return;

    var mv = Math.min(5000,
      Math.max.apply(null, vel.concat(thr)) * 1.2);
    var p = 20;

    function dl(a, c) {
      x.beginPath();
      for (var i = 0; i < ts.length; i++) {
        var px = p + (ts[i] - t0) / wd * (w - 2*p);
        var py = h - p - Math.min(a[i], mv) / mv * (h - 2*p);
        if (i === 0) x.moveTo(px, py); else x.lineTo(px, py);
      }
      x.strokeStyle = c; x.lineWidth = 1.5; x.stroke();
    }

    dl(vel, '#ff9800');
    dl(thr, '#e91e63');
    dl(fv, '#00bcd4');

    x.fillStyle = '#888'; x.font = '11px sans-serif';
    x.fillText('Velocity (px/s)', 5, 14);
    x.fillText('0', 5, h - 5);
    x.fillText(Math.round(mv) + '', 5, 25);
  });
}

function drawClass() {
  var d = D[ai];
  dc('cC', function(x, w, h) {
    var wi = gwi(d), i0 = wi[0], i1 = wi[1], t0 = wi[2];
    var ts = d.timestamps.slice(i0, i1 + 1);
    var gt = d.gt_labels.slice(i0, i1 + 1);
    var pr = d.pred_labels.slice(i0, i1 + 1);
    var p = 20, rh = (h - 2*p) / 3;

    for (var i = 0; i < ts.length; i++) {
      var px = p + (ts[i] - t0) / wd * (w - 2*p);
      var bw = Math.max(2, (w - 2*p) / ts.length);
      x.fillStyle = gt[i] === 1 ? '#4caf50' : '#f44336';
      x.fillRect(px, p, bw, rh - 2);
      x.fillStyle = pr[i] === 1 ? '#4caf50' : '#f44336';
      x.fillRect(px, p + rh, bw, rh - 2);
      x.fillStyle = gt[i] === pr[i]
        ? 'rgba(76,175,80,0.27)' : 'rgba(244,67,54,0.53)';
      x.fillRect(px, p + 2*rh, bw, rh - 2);
    }

    x.fillStyle = '#888'; x.font = '11px sans-serif';
    x.fillText('Ground Truth', 5, p + rh/2 + 4);
    x.fillText('With Flow', 5, p + rh + rh/2 + 4);
    x.fillText('Match?', 5, p + 2*rh + rh/2 + 4);
  });
}

function drawFlow() {
  var d = D[ai];
  dc('cF', function(x, w, h) {
    var wi = gwi(d), i0 = wi[0], i1 = wi[1], t0 = wi[2];
    var ts = d.timestamps.slice(i0, i1 + 1);
    var gt = d.gt_labels.slice(i0, i1 + 1);
    var pr = d.pred_labels.slice(i0, i1 + 1);
    var pn = d.pred_noflow_labels.slice(i0, i1 + 1);
    var p = 20, rh = (h - 2*p) / 3;

    for (var i = 0; i < ts.length; i++) {
      var px = p + (ts[i] - t0) / wd * (w - 2*p);
      var bw = Math.max(2, (w - 2*p) / ts.length);
      x.fillStyle = gt[i] === 1 ? '#4caf50' : '#f44336';
      x.fillRect(px, p, bw, rh - 2);
      x.fillStyle = pr[i] === 1 ? '#4caf50' : '#f44336';
      x.fillRect(px, p + rh, bw, rh - 2);
      x.fillStyle = pn[i] === 1 ? '#4caf50' : '#f44336';
      x.fillRect(px, p + 2*rh, bw, rh - 2);
    }

    x.fillStyle = '#888'; x.font = '11px sans-serif';
    x.fillText('Ground Truth', 5, p + rh/2 + 4);
    x.fillText('With Flow', 5, p + rh + rh/2 + 4);
    x.fillText('Without Flow', 5, p + 2*rh + rh/2 + 4);
  });
}

/* ── Slider ── */
var sl = document.getElementById('sl');
sl.addEventListener('input', function() {
  var d = D[ai];
  var mt = d.timestamps[d.timestamps.length - 1];
  ws[ai] = (sl.value / 100) * Math.max(0, mt - wd);
  utl();
  drawAll();
});
function utl() {
  var s = ws[ai] || 0;
  document.getElementById('tl').textContent =
    s.toFixed(1) + '-' + (s + wd).toFixed(1) + 's';
}

function drawAll() {
  drawGaze(); drawVel(); drawClass(); drawFlow();
}

window.addEventListener('resize', drawAll);
init();
setTimeout(drawAll, 100);
</script>
</body>
</html>"""

    html = html_template.replace("__DATA_PLACEHOLDER__",
                                 json.dumps(vis_data_list))
    html = html.replace("__RESULTS_PLACEHOLDER__",
                         json.dumps(ok_results))

    output_path = os.path.join(os.path.dirname(__file__), "data",
                               "visualization", "verify_all_datasets.html")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path


# ── Main ──────────────────────────────────────────────────────────────
def main():
    print("=" * 78)
    print("  MULTI-DATASET VERIFICATION: I-VAT+Frel (Head-Mounted Eye Tracking)")
    print("=" * 78)
    print(f"\n  Datasets:   {DATASET_ROOT}")
    print(f"  Resolution: {RESOLUTION}")
    print(f"  Threshold:  {VELOCITY_THRESHOLD} px/ms ({VELOCITY_THRESHOLD*1000:.0f} px/s)")
    print(f"  Savgol:     {SAVGOL_WINDOW_MS} ms")
    print(f"  Gain:       {GAIN}")
    print()

    results = []
    vis_data_list = []

    for category in CATEGORIES:
        for subject in SUBJECTS:
            label = f"{category}/{subject}"
            data_dir = os.path.join(DATASET_ROOT, category, subject)
            if not os.path.isdir(data_dir):
                print(f"  SKIP  {label}  (directory not found)")
                continue

            print(f"  Processing {label} ...", end="", flush=True)
            try:
                r, vis = verify_dataset(category, subject)
                results.append(r)
                vis_data_list.append(vis)
                status = "PASS" if r["vel_pass"] else "FAIL"
                print(f"  {status}  (vel_corr={r['vel_correlation']:.4f}, "
                      f"F1_fix={r['f1_fix_flow']:.4f})")
            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({
                    "category": category, "subject": subject,
                    "error": str(e),
                })

    # ── Summary Table ─────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  RESULTS SUMMARY")
    print("=" * 78)

    # Header
    hdr = (f"{'Dataset':<13} {'Samples':>7} {'Hz':>5} "
           f"{'F1-Fix':>7} {'F1-Sac':>7} "
           f"{'F1-Fix':>7} {'F1-Sac':>7} "
           f"{'F1-Fix':>7} {'F1-Sac':>7} "
           f"{'Improv':>7} {'VelCorr':>9} {'VelOK':>5}")
    sub = (f"{'':13} {'':>7} {'':>5} "
           f"{'(flow)':>7} {'(flow)':>7} "
           f"{'(+50ms)':>7} {'(+50ms)':>7} "
           f"{'(noFlw)':>7} {'(noFlw)':>7} "
           f"{'':>7} {'':>9} {'':>5}")
    print(f"\n{hdr}")
    print(sub)
    print("-" * len(hdr))

    ok_results = [r for r in results if "error" not in r]
    for r in results:
        if "error" in r:
            print(f"  {r['category']}/{r['subject']:<8}  ERROR: {r['error']}")
            continue
        vel_ok = "YES" if r["vel_pass"] else "NO"
        print(f"  {r['category']}/{r['subject']:<8} "
              f"{r['n_samples']:>7} {r['sampling_rate']:>5.0f} "
              f"{r['f1_fix_flow']:>7.4f} {r['f1_sac_flow']:>7.4f} "
              f"{r['f1_fix_flow50']:>7.4f} {r['f1_sac_flow50']:>7.4f} "
              f"{r['f1_fix_noflow']:>7.4f} {r['f1_sac_noflow']:>7.4f} "
              f"{r['f1_improvement']:>+7.4f} {r['vel_correlation']:>9.6f} "
              f"{vel_ok:>5}")

    if not ok_results:
        print("\n  No datasets processed successfully!")
        return

    # ── Aggregate Statistics ──────────────────────────────────────
    print(f"\n{'─'*78}")
    print("  AGGREGATE STATISTICS")
    print(f"{'─'*78}")

    f1s_flow = [r["f1_fix_flow"] for r in ok_results]
    f1s_noflow = [r["f1_fix_noflow"] for r in ok_results]
    vel_corrs = [r["vel_correlation"] for r in ok_results]
    improvements = [r["f1_improvement"] for r in ok_results]

    print(f"\n  F1-Fixation (with flow):    mean={np.mean(f1s_flow):.4f}  "
          f"min={np.min(f1s_flow):.4f}  max={np.max(f1s_flow):.4f}")
    print(f"  F1-Fixation (no flow):      mean={np.mean(f1s_noflow):.4f}  "
          f"min={np.min(f1s_noflow):.4f}  max={np.max(f1s_noflow):.4f}")
    print(f"  Flow improvement:           mean={np.mean(improvements):+.4f}  "
          f"min={np.min(improvements):+.4f}  max={np.max(improvements):+.4f}")
    print(f"  Velocity correlation:       mean={np.mean(vel_corrs):.6f}  "
          f"min={np.min(vel_corrs):.6f}")

    # ── Dynamic vs Static comparison ──────────────────────────────
    print(f"\n{'─'*78}")
    print("  DYNAMIC vs STATIC COMPARISON")
    print(f"{'─'*78}")
    print("  (Dynamic = head motion -> flow compensation should help more)")

    for cat in CATEGORIES:
        cat_results = [r for r in ok_results if r["category"] == cat]
        if not cat_results:
            continue
        cat_flow = [r["f1_fix_flow"] for r in cat_results]
        cat_noflow = [r["f1_fix_noflow"] for r in cat_results]
        cat_improv = [r["f1_improvement"] for r in cat_results]
        print(f"\n  {cat.upper()} ({len(cat_results)} datasets):")
        print(f"    F1 with flow:    mean={np.mean(cat_flow):.4f}  "
              f"[{', '.join(f'{v:.4f}' for v in cat_flow)}]")
        print(f"    F1 without flow: mean={np.mean(cat_noflow):.4f}  "
              f"[{', '.join(f'{v:.4f}' for v in cat_noflow)}]")
        print(f"    Improvement:     mean={np.mean(cat_improv):+.4f}  "
              f"[{', '.join(f'{v:+.4f}' for v in cat_improv)}]")

    # ── Pass/Fail Verdict ─────────────────────────────────────────
    print(f"\n{'='*78}")
    print("  VERDICT")
    print(f"{'='*78}")

    all_vel_pass = all(r["vel_pass"] for r in ok_results)
    all_f1_ok = all(r["f1_fix_flow"] > 0.5 for r in ok_results)
    n_flow_helps = sum(1 for r in ok_results if r["f1_improvement"] >= 0)

    print(f"\n  Velocity computation correct:  "
          f"{'PASS' if all_vel_pass else 'FAIL'}  "
          f"(all correlations > 0.99: {all_vel_pass})")
    print(f"  Classification reasonable:     "
          f"{'PASS' if all_f1_ok else 'FAIL'}  "
          f"(all F1 > 0.5: {all_f1_ok})")
    print(f"  Flow helps detection:          "
          f"{n_flow_helps}/{len(ok_results)} datasets improved with flow")

    if all_vel_pass and all_f1_ok:
        print(f"\n  >>> OVERALL: PASS -- Implementation is correct across "
              f"all {len(ok_results)} datasets <<<")
    else:
        failed = [f"{r['category']}/{r['subject']}" for r in ok_results
                  if not r["vel_pass"] or r["f1_fix_flow"] <= 0.5]
        print(f"\n  >>> OVERALL: ISSUES DETECTED in: {', '.join(failed)} <<<")

    # ── Generate HTML report ──────────────────────────────────────
    if ok_results and vis_data_list:
        html_path = generate_html(ok_results, vis_data_list)
        print(f"\n  HTML report saved to: {html_path}")
        print(f"  -> Open in browser to see interactive charts for all datasets")

    print()


if __name__ == "__main__":
    main()
