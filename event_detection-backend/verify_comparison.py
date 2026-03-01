"""
=============================================================================
 COMPARISON VERIFICATION: Your Code vs Lukas-Wilde Thesis I-VT / I-VAT+Frel
=============================================================================

Runs both implementations on ALL 8 datasets (dynamic/static × A-D) and
compares F1 scores, velocity signals, and per-sample classification labels
between:

  1. **Your code** – EventDetection + classify_ivt  (src/)
  2. **Lukas thesis** – SavgolFilter → VelocityCalculator /
     RelativeVelocityCalculator → (Adaptive)Threshold → classify  (imported)

For each dataset it reports:
  - Sample-level F1 (Fixation / Saccade) for both implementations
  - Per-sample agreement (% identical labels)
  - Velocity Pearson correlation between the two pipelines
  - Absolute F1 difference

HOW TO RUN:
  cd event_detection-backend
  python verify_comparison.py
=============================================================================
"""

import json
import os
import sys
import types as _types
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

# ── Your code imports ─────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from src import EventDetection
from src.detection_algorithms import classify_ivt
from src.preprocessing import (
    apply_savgol_filter, compute_gaze_velocity, compute_flow_velocity,
    compute_relative_velocity, compute_flow_window_rms,
    compute_adaptive_threshold,
)

# ── Lukas thesis imports ──────────────────────────────────────────────
# The thesis code uses bare imports (e.g. 'from transformations import ...',
# 'from datasets import ...') that require specific directories on sys.path.
# The transformations/__init__.py imports ALL transformation classes including
# heavy-dependency ones (torch, dask).  We pre-register a lightweight package
# so only the submodules we need are loaded.
THESIS_ROOT = os.path.join(os.path.dirname(__file__), "lukas-wilde-eyetracking-thesis")
for _p in [THESIS_ROOT,
           os.path.join(THESIS_ROOT, "algorithms"),
           os.path.join(THESIS_ROOT, "datasets")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

_tf_pkg = _types.ModuleType("transformations")
_tf_pkg.__path__ = [os.path.join(THESIS_ROOT, "algorithms", "transformations")]
_tf_pkg.__package__ = "transformations"
sys.modules.setdefault("transformations", _tf_pkg)

from transformations.savgol_filter import SavgolFilter as LukasSavgolFilter
from transformations.velocity_calculator import (
    VelocityCalculator as LukasVelocityCalculator,
    RelativeVelocityCalculator as LukasRelativeVelocityCalculator,
)
from transformations.adaptive_threshold import (
    AdaptiveThreshold as LukasAdaptiveThreshold,
)
from event_matching.event import EventType
from gaze_dataset import GazeDataset as _LukasGazeDatasetBase, KeyParams

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

# Lukas thesis uses px/s thresholds (1200 = 1.2 px/ms × 1000)
LUKAS_THRESHOLD_PXS = 1200         # px/s  – default in IVTParams
LUKAS_THRESHOLD_PXMS = LUKAS_THRESHOLD_PXS / 1000.0  # 1.2 px/ms
LUKAS_GAIN = 1.0
LUKAS_WINDOW_SIZE_MS = 500

DATASET_ROOT = os.path.join(os.path.dirname(__file__), "dataset")
CATEGORIES = ["dynamic", "static"]
SUBJECTS = ["A", "B", "C", "D"]


# ── Minimal GazeDataset wrapper for thesis pipeline ───────────────────

@dataclass(frozen=True)
class _SingleKey(KeyParams):
    name: str


class _SingleSplitDataset(_LukasGazeDatasetBase[_SingleKey]):
    """Wraps a single DataFrame in the GazeDataset interface so that
    the real thesis Transformation classes can operate on it."""

    def __init__(self, df: pd.DataFrame, *, sample_rate_hz: int):
        self._sr = sample_rate_hz
        super().__init__(splits={_SingleKey(name="single"): df})

    @property
    def name(self): return "verify"

    @property
    def sample_rate_hz(self): return self._sr

    @property
    def width(self): return RESOLUTION[0]

    @property
    def height(self): return RESOLUTION[1]

    # Abstract methods we never call — satisfy the ABC
    def get_flow_path(self, p):           raise NotImplementedError
    def get_video_path(self, p):          raise NotImplementedError
    def get_gaze_path(self, p):           raise NotImplementedError
    def get_similarity_path(self, p):     raise NotImplementedError
    def get_time_path(self, p):           raise NotImplementedError
    def get_label_path(self, p):          raise NotImplementedError
    def _load_gaze(self, path):           raise NotImplementedError
    @staticmethod
    def _load_timestamps(path):           raise NotImplementedError
    @staticmethod
    def _load_labels(path):               raise NotImplementedError
    def _load_all_splits(self):           pass
    def get_train_test_for_fold(self, f): raise NotImplementedError


# ── Helpers ───────────────────────────────────────────────────────────

def load_dataset(data_dir):
    """Load .npy files from a dataset folder and return raw arrays."""
    gaze = np.load(os.path.join(data_dir, "gaze.npy"))
    time_gaze = np.load(os.path.join(data_dir, "time_gaze.npy"))
    gt_labels = np.load(os.path.join(data_dir, "gt_labels.npy"))
    optic_flow = np.load(os.path.join(data_dir, "optic_flow.npy"))
    time_of = np.load(os.path.join(data_dir, "time_optic_flow.npy"))
    time_sc = np.load(os.path.join(data_dir, "time_scene_camera.npy"))
    return gaze, time_gaze, gt_labels, optic_flow, time_of, time_sc


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


def prepare_lukas_dataframe(gaze, time_gaze, optic_flow, time_of, time_sc):
    """Build a DataFrame mimicking the Lukas thesis DrewsDataset loader.

    The thesis code:
      1. Loads gaze (x, y), timestamps, and gt_labels into a DataFrame.
      2. Computes video_timestamp by merge_asof with scene-camera times.
      3. OpticFlowLoader: computes mean flow per frame, assigns to gaze
         samples via the 'frame' column.

    We replicate this logic here.
    """
    # Build gaze DataFrame with timestamps
    gaze_df = pd.DataFrame({
        "x": gaze[:, 0],
        "y": gaze[:, 1],
        "timestamp": time_gaze,
    })

    # Merge with video timestamps (like gaze_dataset._load_split)
    video_df = pd.DataFrame({
        "video_timestamp": time_sc,
        "frame": np.arange(len(time_sc)),
    })
    merged = pd.merge_asof(
        gaze_df,
        video_df,
        left_on="timestamp",
        right_on="video_timestamp",
        direction="backward",
    )
    merged = merged.fillna(value={"frame": 0})

    # OpticFlowLoader: compute mean flow per frame, assign by frame index
    mean_flow = np.nanmean(optic_flow.astype(np.float64), axis=(1, 2))
    frames = merged["frame"].astype(int).values
    frames = np.clip(frames, 0, mean_flow.shape[0] - 1)
    merged["flow_x"] = mean_flow[frames, 0]
    merged["flow_y"] = mean_flow[frames, 1]

    return merged


# ── Lukas Thesis Pipeline (using real thesis transformation classes) ──

def lukas_classify_ivt(df, use_flow, adaptive, sampling_rate):
    """
    Run the Lukas thesis I-VT classification on a DataFrame.

    Uses the REAL thesis transformation classes (SavgolFilter,
    VelocityCalculator, RelativeVelocityCalculator, AdaptiveThreshold)
    imported from lukas-wilde-eyetracking-thesis/.

    Returns per-sample binary array (1=Fixation, 0=Saccade) and the
    velocity column used for classification.
    """
    sr = int(round(sampling_rate))
    dataset = _SingleSplitDataset(df, sample_rate_hz=sr)

    # Step 1: Savgol filter  (thesis SavgolFilter class)
    LukasSavgolFilter(55, sr).apply(dataset)

    if use_flow:
        # Steps 2-4: gaze velocity + flow velocity + relative velocity
        # (thesis RelativeVelocityCalculator calls VelocityCalculator.apply
        #  internally, then add_flow_velocity, then subtracts)
        LukasRelativeVelocityCalculator(
            x_col="filter_x", y_col="filter_y"
        ).apply(dataset)
        mag_col = "vel_rel_mag"

        if adaptive:
            # Step 5: Adaptive threshold (thesis AdaptiveThreshold class)
            window_size = int(
                LUKAS_WINDOW_SIZE_MS // dataset.sample_duration_ms
            )
            LukasAdaptiveThreshold(
                window_size, LUKAS_GAIN, LUKAS_THRESHOLD_PXS
            ).apply(dataset)

            _, split = dataset.get_splits()[0]
            pred = np.where(
                split[mag_col] < split["threshold"],
                1,  # Fixation
                0,  # Saccade
            )
        else:
            _, split = dataset.get_splits()[0]
            pred = np.where(
                split[mag_col] < LUKAS_THRESHOLD_PXS, 1, 0,
            )
    else:
        # No flow: standard gaze velocity only
        LukasVelocityCalculator(
            x_col="filter_x", y_col="filter_y"
        ).apply(dataset)
        mag_col = "vel_mag"

        _, split = dataset.get_splits()[0]
        pred = np.where(
            split[mag_col] < LUKAS_THRESHOLD_PXS, 1, 0,
        )

    return pred, split[mag_col].values


# ── Per-dataset comparison ────────────────────────────────────────────

def compare_dataset(category, subject):
    """Run both pipelines on one dataset and compare. Returns metrics + vis."""
    data_dir = os.path.join(DATASET_ROOT, category, subject)
    label = f"{category}/{subject}"

    gaze, time_gaze, gt_labels, optic_flow, time_of, time_sc = load_dataset(data_dir)
    N = len(gaze)
    sampling_rate = 1.0 / np.median(np.diff(time_gaze))

    # ── YOUR CODE (with flow, adaptive) ───────────────────────────
    df_yours = prepare_dataframe(gaze, time_gaze, optic_flow, time_of)
    det = EventDetection(
        df_yours.copy(), resolution=RESOLUTION,
        column_mapping=COLUMN_MAPPING_FLOW, is_normalized=False,
    )
    result_yours, _ = classify_ivt(
        det.gaze_data.copy(),
        velocity_threshold=VELOCITY_THRESHOLD,
        min_fixation_duration=0,
        adapt=True,
        sampling_rate=sampling_rate,
    )
    pred_yours = (result_yours["event_type"] == "Fixation").astype(int).values

    # YOUR CODE (no flow)
    df_yours_nf = df_yours.drop(columns=["flow_x", "flow_y", "video_timestamp"])
    det_nf = EventDetection(
        df_yours_nf.copy(), resolution=RESOLUTION,
        column_mapping=COLUMN_MAPPING_NO_FLOW, is_normalized=False,
    )
    result_yours_nf, _ = classify_ivt(
        det_nf.gaze_data.copy(),
        velocity_threshold=VELOCITY_THRESHOLD,
        min_fixation_duration=0,
        adapt=False,
    )
    pred_yours_nf = (result_yours_nf["event_type"] == "Fixation").astype(int).values

    # ── LUKAS THESIS (with flow, adaptive) ────────────────────────
    df_lukas = prepare_lukas_dataframe(gaze, time_gaze, optic_flow, time_of, time_sc)
    pred_lukas_flow, vel_lukas_flow = lukas_classify_ivt(
        df_lukas.copy(), use_flow=True, adaptive=True, sampling_rate=sampling_rate,
    )

    # LUKAS THESIS (no flow)
    df_lukas_nf = prepare_lukas_dataframe(gaze, time_gaze, optic_flow, time_of, time_sc)
    pred_lukas_nf, vel_lukas_nf = lukas_classify_ivt(
        df_lukas_nf.copy(), use_flow=False, adaptive=False, sampling_rate=sampling_rate,
    )

    # ── F1 scores (vs ground truth) ──────────────────────────────
    f1_yours_fix = f1_score(gt_labels, pred_yours, pos_label=1)
    f1_yours_sac = f1_score(gt_labels, pred_yours, pos_label=0)
    f1_yours_nf_fix = f1_score(gt_labels, pred_yours_nf, pos_label=1)
    f1_yours_nf_sac = f1_score(gt_labels, pred_yours_nf, pos_label=0)

    f1_lukas_fix = f1_score(gt_labels, pred_lukas_flow, pos_label=1)
    f1_lukas_sac = f1_score(gt_labels, pred_lukas_flow, pos_label=0)
    f1_lukas_nf_fix = f1_score(gt_labels, pred_lukas_nf, pos_label=1)
    f1_lukas_nf_sac = f1_score(gt_labels, pred_lukas_nf, pos_label=0)

    # ── Per-sample agreement ─────────────────────────────────────
    agree_flow = np.mean(pred_yours == pred_lukas_flow) * 100
    agree_noflow = np.mean(pred_yours_nf == pred_lukas_nf) * 100

    # ── Velocity correlation ──────────────────────────────────────
    # Compare gaze velocity signals between the two pipelines.
    # Your code stores vel_mag in px/ms, Lukas in px/s  → convert yours.
    yours_vel = result_yours["vel_mag"].values * 1000  # → px/s
    lukas_vel = vel_lukas_flow  # already px/s (relative when with-flow)
    # For fair comparison of gaze velocity (before flow subtraction):
    yours_gaze_vel = result_yours["vel_mag"].values * 1000  # px/s
    # Lukas gaze velocity (before flow subtraction) — use real thesis classes
    df_lukas_v = prepare_lukas_dataframe(gaze, time_gaze, optic_flow, time_of, time_sc)
    ds_v = _SingleSplitDataset(df_lukas_v, sample_rate_hz=int(round(sampling_rate)))
    LukasSavgolFilter(55, int(round(sampling_rate))).apply(ds_v)
    LukasVelocityCalculator(x_col="filter_x", y_col="filter_y").apply(ds_v)
    _, split_v = ds_v.get_splits()[0]
    lukas_gaze_vel = split_v["vel_mag"].values  # px/s

    mask = np.isfinite(yours_gaze_vel) & np.isfinite(lukas_gaze_vel)
    if mask.sum() > 2:
        vel_corr = np.corrcoef(yours_gaze_vel[mask], lukas_gaze_vel[mask])[0, 1]
    else:
        vel_corr = float("nan")

    # ── Prepare visualization data ────────────────────────────────
    sim_df = det.gaze_data.copy()
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
        "pred_yours": pred_yours[::step].tolist(),
        "pred_yours_nf": pred_yours_nf[::step].tolist(),
        "pred_lukas": pred_lukas_flow[::step].tolist(),
        "pred_lukas_nf": pred_lukas_nf[::step].tolist(),
        "velocity_yours": np.nan_to_num(
            sim_df["vel_rel_mag"].values[::step] * 1000).tolist(),
        "threshold_yours": np.nan_to_num(
            sim_df["threshold"].values[::step] * 1000).tolist(),
        "flow_vel": np.nan_to_num(
            sim_df["flow_vel_mag"].values[::step] * 1000).tolist(),
    }

    # Add Lukas velocity & threshold for comparison — use real thesis classes
    df_lukas_vis = prepare_lukas_dataframe(gaze, time_gaze, optic_flow, time_of, time_sc)
    sr_int = int(round(sampling_rate))
    ds_vis = _SingleSplitDataset(df_lukas_vis, sample_rate_hz=sr_int)
    LukasSavgolFilter(55, sr_int).apply(ds_vis)
    LukasRelativeVelocityCalculator(
        x_col="filter_x", y_col="filter_y"
    ).apply(ds_vis)
    window_size_vis = int(LUKAS_WINDOW_SIZE_MS // ds_vis.sample_duration_ms)
    LukasAdaptiveThreshold(
        window_size_vis, LUKAS_GAIN, LUKAS_THRESHOLD_PXS
    ).apply(ds_vis)
    _, split_vis = ds_vis.get_splits()[0]
    vis_data["velocity_lukas"] = np.nan_to_num(
        split_vis["vel_rel_mag"].values[::step]).tolist()
    vis_data["threshold_lukas"] = np.nan_to_num(
        split_vis["threshold"].values[::step]).tolist()

    metrics = {
        "category": category,
        "subject": subject,
        "n_samples": int(N),
        "sampling_rate": round(float(sampling_rate), 1),
        # Your code
        "f1_yours_fix": round(float(f1_yours_fix), 4),
        "f1_yours_sac": round(float(f1_yours_sac), 4),
        "f1_yours_nf_fix": round(float(f1_yours_nf_fix), 4),
        "f1_yours_nf_sac": round(float(f1_yours_nf_sac), 4),
        # Lukas thesis
        "f1_lukas_fix": round(float(f1_lukas_fix), 4),
        "f1_lukas_sac": round(float(f1_lukas_sac), 4),
        "f1_lukas_nf_fix": round(float(f1_lukas_nf_fix), 4),
        "f1_lukas_nf_sac": round(float(f1_lukas_nf_sac), 4),
        # Comparison
        "agree_flow": round(float(agree_flow), 2),
        "agree_noflow": round(float(agree_noflow), 2),
        "vel_correlation": round(float(vel_corr), 6),
        # Deltas
        "f1_delta_flow": round(float(f1_yours_fix - f1_lukas_fix), 4),
        "f1_delta_noflow": round(float(f1_yours_nf_fix - f1_lukas_nf_fix), 4),
    }

    return metrics, vis_data


# ── HTML Report ───────────────────────────────────────────────────────

def generate_html(results, vis_data_list):
    """Generate interactive HTML comparison report."""

    html_template = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Code Comparison: Your Code vs Lukas Thesis</title>
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
  .verdict.warn { background: #ff980033; color: #ff9800;
                  border: 2px solid #ff9800; }
  .verdict.fail { background: #b71c1c33; color: #f44336;
                  border: 2px solid #f44336; }
  .stats { display: flex; gap: 12px; flex-wrap: wrap; margin: 12px 0; }
  .sb { background: #16213e; padding: 12px 16px; border-radius: 8px;
        border-left: 4px solid #6dd5fa; min-width: 150px; flex: 1; }
  .sb .v { font-size: 1.5em; font-weight: bold; color: #6dd5fa; }
  .sb .l { font-size: .8em; color: #888; margin-top: 3px; }
  .sb.g .v { color: #4caf50; }
  .sb.b .v { color: #f44336; }
  .sb.y .v { color: #ff9800; }
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
  .yt { color: #ff9800; }
  .info { background: #16213e; padding: 12px; border-radius: 8px;
          margin: 8px 0; line-height: 1.5; font-size: .9em; }
  .note { background: #1a237e33; padding: 10px; border-radius: 6px;
          border-left: 3px solid #6dd5fa; margin: 8px 0; font-size: .9em; }
</style>
</head>
<body>

<h1>Code Comparison: Your Code vs Lukas Thesis</h1>
<p class="sub">Comparing I-VT / I-VAT+Frel implementations across 8
  datasets (dynamic/static &times; A-D)</p>

<div class="note">
  <strong>Note:</strong> Your code uses threshold=<strong>1.0 px/ms</strong>
  (1000 px/s). Lukas thesis default=<strong>1200 px/s</strong> (1.2 px/ms).
  Differences in F1 are expected due to this threshold gap, plus minor
  pipeline differences (data loading, flow mapping, segment handling).
  <br>The key comparison is <em>velocity correlation</em> and
  <em>classification agreement</em>.
</div>

<div class="verdict" id="vb"></div>
<div class="stats" id="as"></div>

<h2>Summary Table</h2>
<p class="sub">Click a row to jump to that dataset's charts</p>
<table id="stbl">
  <thead><tr>
    <th>Dataset</th><th>Samples</th><th>Hz</th>
    <th>F1-Fix<br>(Yours,flow)</th>
    <th>F1-Fix<br>(Lukas,flow)</th>
    <th>F1 &Delta;</th>
    <th>F1-Sac<br>(Yours,flow)</th>
    <th>F1-Sac<br>(Lukas,flow)</th>
    <th>F1 &Delta;</th>
    <th>F1-Fix<br>(Yours,noFlow)</th>
    <th>F1-Fix<br>(Lukas,noFlow)</th>
    <th>F1 &Delta;</th>
    <th>Agreement<br>(flow)</th>
    <th>Vel Corr</th>
  </tr></thead>
  <tbody></tbody>
</table>

<h2>Dataset Details</h2>
<div class="tabs" id="tbs"></div>
<div class="tc" id="tc">
  <div class="stats" id="ds"></div>

  <h2>Ground Truth vs Both Predictions (with flow)</h2>
  <div class="cc">
    <canvas id="cC" height="160"></canvas>
    <div class="leg">
      <span><span class="d" style="background:#4caf50"></span>
        Fixation</span>
      <span><span class="d" style="background:#f44336"></span>
        Saccade</span>
    </div>
  </div>

  <h2>Your Code: With Flow vs Without Flow</h2>
  <div class="cc">
    <canvas id="cY" height="120"></canvas>
  </div>

  <h2>Lukas Thesis: With Flow vs Without Flow</h2>
  <div class="cc">
    <canvas id="cL" height="120"></canvas>
  </div>

  <h2>Agreement Map (with flow)</h2>
  <div class="cc">
    <canvas id="cA" height="80"></canvas>
    <div class="leg">
      <span><span class="d" style="background:#4caf50"></span>
        Both agree</span>
      <span><span class="d" style="background:#f44336"></span>
        Disagree</span>
    </div>
  </div>

  <h2>Velocity &amp; Threshold Comparison</h2>
  <div class="cc">
    <div class="leg">
      <span><span class="d" style="background:#ff9800"></span>
        Your relative vel (px/s)</span>
      <span><span class="d" style="background:#e91e63"></span>
        Your threshold</span>
      <span><span class="d" style="background:#2196f3"></span>
        Lukas relative vel (px/s)</span>
      <span><span class="d" style="background:#9c27b0"></span>
        Lukas threshold</span>
      <span><span class="d" style="background:#00bcd4"></span>
        Camera flow vel</span>
    </div>
    <canvas id="cV" height="220"></canvas>
    <div class="info">
      <strong>How to read:</strong> When velocity is BELOW the threshold
      &rarr; Fixation. Above &rarr; Saccade.
      Note: Your threshold is ~1000 px/s base, Lukas is ~1200 px/s base.
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
  var c = g === true ? 'g' : (g === false ? 'b' : (g === 'warn' ? 'y' : ''));
  return '<div class="sb ' + c + '"><div class="v">' + v +
         '</div><div class="l">' + l + '</div></div>';
}

function pm(v) { return (v >= 0 ? '+' : '') + v; }

function init() {
  var mc = avg(R.map(function(r) { return r.vel_correlation; }));
  var ma = avg(R.map(function(r) { return r.agree_flow; }));
  var high_agree = ma > 90;
  var high_corr = mc > 0.95;

  var vb = document.getElementById('vb');
  if (high_agree && high_corr) {
    vb.className = 'verdict pass';
    vb.textContent = 'IMPLEMENTATIONS MATCH WELL (agreement ' +
      ma.toFixed(1) + '%, vel corr ' + mc.toFixed(4) + ')';
  } else if (ma > 80) {
    vb.className = 'verdict warn';
    vb.textContent = 'PARTIAL MATCH (agreement ' + ma.toFixed(1) +
      '%) — threshold difference is the main factor';
  } else {
    vb.className = 'verdict fail';
    vb.textContent = 'SIGNIFICANT DIFFERENCES DETECTED';
  }

  var mfy = avg(R.map(function(r) { return r.f1_yours_fix; }));
  var mfl = avg(R.map(function(r) { return r.f1_lukas_fix; }));
  var mfd = avg(R.map(function(r) { return r.f1_delta_flow; }));
  var msy = avg(R.map(function(r) { return r.f1_yours_sac; }));
  var msl = avg(R.map(function(r) { return r.f1_lukas_sac; }));

  document.getElementById('as').innerHTML = [
    sb(mfy.toFixed(4), 'Mean F1-Fix (Yours)', mfy > 0.8),
    sb(mfl.toFixed(4), 'Mean F1-Fix (Lukas)', mfl > 0.8),
    sb(msy.toFixed(4), 'Mean F1-Sac (Yours)', msy > 0.5),
    sb(msl.toFixed(4), 'Mean F1-Sac (Lukas)', msl > 0.5),
    sb(pm(mfd.toFixed(4)), 'Mean F1-Fix Delta', Math.abs(mfd) < 0.05 ? true : 'warn'),
    sb(ma.toFixed(1) + '%', 'Mean Agreement', high_agree),
    sb(mc.toFixed(4), 'Mean Vel Corr', high_corr),
    sb(R.length, 'Datasets', null)
  ].join('');

  var tb = document.querySelector('#stbl tbody');
  R.forEach(function(r, i) {
    var tr = document.createElement('tr');
    tr.className = 'ck';
    tr.onclick = function() { sel(i); };
    var dc = Math.abs(r.f1_delta_flow) < 0.05 ? 'pt' :
             (Math.abs(r.f1_delta_flow) < 0.1 ? 'yt' : 'ft');
    var sc = Math.abs(r.f1_yours_sac - r.f1_lukas_sac) < 0.05 ? 'pt' :
             (Math.abs(r.f1_yours_sac - r.f1_lukas_sac) < 0.1 ? 'yt' : 'ft');
    tr.innerHTML =
      '<td>' + r.category + '/' + r.subject + '</td>' +
      '<td>' + r.n_samples + '</td><td>' + r.sampling_rate + '</td>' +
      '<td>' + r.f1_yours_fix + '</td>' +
      '<td>' + r.f1_lukas_fix + '</td>' +
      '<td class="' + dc + '">' + pm(r.f1_delta_flow) + '</td>' +
      '<td>' + r.f1_yours_sac + '</td>' +
      '<td>' + r.f1_lukas_sac + '</td>' +
      '<td class="' + sc + '">' + pm((r.f1_yours_sac - r.f1_lukas_sac).toFixed(4)) + '</td>' +
      '<td>' + r.f1_yours_nf_fix + '</td>' +
      '<td>' + r.f1_lukas_nf_fix + '</td>' +
      '<td class="' + dc + '">' + pm(r.f1_delta_noflow) + '</td>' +
      '<td>' + r.agree_flow + '%</td>' +
      '<td>' + r.vel_correlation + '</td>';
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
  var dc = Math.abs(r.f1_delta_flow) < 0.05 ? true :
           (Math.abs(r.f1_delta_flow) < 0.1 ? 'warn' : false);
  document.getElementById('ds').innerHTML = [
    sb(r.f1_yours_fix, 'F1-Fix (Yours)', r.f1_yours_fix > 0.85),
    sb(r.f1_lukas_fix, 'F1-Fix (Lukas)', r.f1_lukas_fix > 0.85),
    sb(r.f1_yours_sac, 'F1-Sac (Yours)', r.f1_yours_sac > 0.5),
    sb(r.f1_lukas_sac, 'F1-Sac (Lukas)', r.f1_lukas_sac > 0.5),
    sb(pm(r.f1_delta_flow), 'F1-Fix Delta', dc),
    sb(r.agree_flow + '%', 'Agreement (flow)', r.agree_flow > 90),
    sb(r.vel_correlation, 'Vel Corr', r.vel_correlation > 0.95),
    sb(r.n_samples, 'Samples', null)
  ].join('');

  document.getElementById('dm').innerHTML =
    '<tr><th>Metric</th><th>Your Code</th><th>Lukas Thesis</th>' +
    '<th>Delta</th></tr>' +
    '<tr><td>F1-Fix (flow, adaptive)</td><td>' +
      r.f1_yours_fix + '</td><td>' + r.f1_lukas_fix + '</td><td>' +
      pm(r.f1_delta_flow) + '</td></tr>' +
    '<tr><td>F1-Sac (flow, adaptive)</td><td>' +
      r.f1_yours_sac + '</td><td>' + r.f1_lukas_sac + '</td><td>' +
      pm((r.f1_yours_sac - r.f1_lukas_sac).toFixed(4)) + '</td></tr>' +
    '<tr><td>F1-Fix (no flow)</td><td>' +
      r.f1_yours_nf_fix + '</td><td>' + r.f1_lukas_nf_fix + '</td><td>' +
      pm(r.f1_delta_noflow) + '</td></tr>' +
    '<tr><td>F1-Sac (no flow)</td><td>' +
      r.f1_yours_nf_sac + '</td><td>' + r.f1_lukas_nf_sac + '</td><td>' +
      pm((r.f1_yours_nf_sac - r.f1_lukas_nf_sac).toFixed(4)) + '</td></tr>' +
    '<tr><td>Agreement (flow)</td><td colspan="2">' +
      r.agree_flow + '%</td><td></td></tr>' +
    '<tr><td>Agreement (no flow)</td><td colspan="2">' +
      r.agree_noflow + '%</td><td></td></tr>' +
    '<tr><td>Velocity correlation</td><td colspan="2">' +
      r.vel_correlation + '</td><td></td></tr>' +
    '<tr><td>Samples</td><td colspan="2">' +
      r.n_samples + '</td><td></td></tr>' +
    '<tr><td>Sampling rate</td><td colspan="2">' +
      r.sampling_rate + ' Hz</td><td></td></tr>';

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

/* Classification: GT, Yours, Lukas, Match */
function drawClass() {
  var d = D[ai];
  dc('cC', function(x, w, h) {
    var wi = gwi(d), i0 = wi[0], i1 = wi[1], t0 = wi[2];
    var ts = d.timestamps.slice(i0, i1 + 1);
    var gt = d.gt_labels.slice(i0, i1 + 1);
    var py = d.pred_yours.slice(i0, i1 + 1);
    var pl = d.pred_lukas.slice(i0, i1 + 1);
    var p = 20, rh = (h - 2*p) / 4;

    for (var i = 0; i < ts.length; i++) {
      var px = p + (ts[i] - t0) / wd * (w - 2*p);
      var bw = Math.max(2, (w - 2*p) / ts.length);
      x.fillStyle = gt[i] === 1 ? '#4caf50' : '#f44336';
      x.fillRect(px, p, bw, rh - 2);
      x.fillStyle = py[i] === 1 ? '#4caf50' : '#f44336';
      x.fillRect(px, p + rh, bw, rh - 2);
      x.fillStyle = pl[i] === 1 ? '#4caf50' : '#f44336';
      x.fillRect(px, p + 2*rh, bw, rh - 2);
      x.fillStyle = gt[i] === py[i] && gt[i] === pl[i]
        ? 'rgba(76,175,80,0.27)' : 'rgba(244,67,54,0.53)';
      x.fillRect(px, p + 3*rh, bw, rh - 2);
    }

    x.fillStyle = '#888'; x.font = '11px sans-serif';
    x.fillText('Ground Truth', 5, p + rh/2 + 4);
    x.fillText('Your Code', 5, p + rh + rh/2 + 4);
    x.fillText('Lukas Thesis', 5, p + 2*rh + rh/2 + 4);
    x.fillText('All Match?', 5, p + 3*rh + rh/2 + 4);
  });
}

/* Your Code: flow vs no-flow */
function drawYours() {
  var d = D[ai];
  dc('cY', function(x, w, h) {
    var wi = gwi(d), i0 = wi[0], i1 = wi[1], t0 = wi[2];
    var ts = d.timestamps.slice(i0, i1 + 1);
    var gt = d.gt_labels.slice(i0, i1 + 1);
    var pf = d.pred_yours.slice(i0, i1 + 1);
    var pn = d.pred_yours_nf.slice(i0, i1 + 1);
    var p = 20, rh = (h - 2*p) / 3;

    for (var i = 0; i < ts.length; i++) {
      var px = p + (ts[i] - t0) / wd * (w - 2*p);
      var bw = Math.max(2, (w - 2*p) / ts.length);
      x.fillStyle = gt[i] === 1 ? '#4caf50' : '#f44336';
      x.fillRect(px, p, bw, rh - 2);
      x.fillStyle = pf[i] === 1 ? '#4caf50' : '#f44336';
      x.fillRect(px, p + rh, bw, rh - 2);
      x.fillStyle = pn[i] === 1 ? '#4caf50' : '#f44336';
      x.fillRect(px, p + 2*rh, bw, rh - 2);
    }

    x.fillStyle = '#888'; x.font = '11px sans-serif';
    x.fillText('Ground Truth', 5, p + rh/2 + 4);
    x.fillText('Yours + Flow', 5, p + rh + rh/2 + 4);
    x.fillText('Yours No Flow', 5, p + 2*rh + rh/2 + 4);
  });
}

/* Lukas: flow vs no-flow */
function drawLukas() {
  var d = D[ai];
  dc('cL', function(x, w, h) {
    var wi = gwi(d), i0 = wi[0], i1 = wi[1], t0 = wi[2];
    var ts = d.timestamps.slice(i0, i1 + 1);
    var gt = d.gt_labels.slice(i0, i1 + 1);
    var pf = d.pred_lukas.slice(i0, i1 + 1);
    var pn = d.pred_lukas_nf.slice(i0, i1 + 1);
    var p = 20, rh = (h - 2*p) / 3;

    for (var i = 0; i < ts.length; i++) {
      var px = p + (ts[i] - t0) / wd * (w - 2*p);
      var bw = Math.max(2, (w - 2*p) / ts.length);
      x.fillStyle = gt[i] === 1 ? '#4caf50' : '#f44336';
      x.fillRect(px, p, bw, rh - 2);
      x.fillStyle = pf[i] === 1 ? '#4caf50' : '#f44336';
      x.fillRect(px, p + rh, bw, rh - 2);
      x.fillStyle = pn[i] === 1 ? '#4caf50' : '#f44336';
      x.fillRect(px, p + 2*rh, bw, rh - 2);
    }

    x.fillStyle = '#888'; x.font = '11px sans-serif';
    x.fillText('Ground Truth', 5, p + rh/2 + 4);
    x.fillText('Lukas + Flow', 5, p + rh + rh/2 + 4);
    x.fillText('Lukas No Flow', 5, p + 2*rh + rh/2 + 4);
  });
}

/* Agreement map */
function drawAgree() {
  var d = D[ai];
  dc('cA', function(x, w, h) {
    var wi = gwi(d), i0 = wi[0], i1 = wi[1], t0 = wi[2];
    var ts = d.timestamps.slice(i0, i1 + 1);
    var py = d.pred_yours.slice(i0, i1 + 1);
    var pl = d.pred_lukas.slice(i0, i1 + 1);
    var p = 20;

    for (var i = 0; i < ts.length; i++) {
      var px = p + (ts[i] - t0) / wd * (w - 2*p);
      var bw = Math.max(2, (w - 2*p) / ts.length);
      x.fillStyle = py[i] === pl[i]
        ? 'rgba(76,175,80,0.6)' : 'rgba(244,67,54,0.8)';
      x.fillRect(px, p, bw, h - 2*p);
    }

    x.fillStyle = '#888'; x.font = '11px sans-serif';
    x.fillText('Agree / Disagree', 5, 14);
  });
}

/* Velocity comparison */
function drawVel() {
  var d = D[ai];
  dc('cV', function(x, w, h) {
    var wi = gwi(d), i0 = wi[0], i1 = wi[1], t0 = wi[2];
    var ts = d.timestamps.slice(i0, i1 + 1);
    var vy = d.velocity_yours.slice(i0, i1 + 1);
    var ty = d.threshold_yours.slice(i0, i1 + 1);
    var vl = d.velocity_lukas.slice(i0, i1 + 1);
    var tl = d.threshold_lukas.slice(i0, i1 + 1);
    var fv = d.flow_vel.slice(i0, i1 + 1);
    if (ts.length < 2) return;

    var all = vy.concat(ty).concat(vl).concat(tl);
    var mv = Math.min(5000,
      Math.max.apply(null, all) * 1.2);
    var p = 20;

    function dl(a, c, lw) {
      x.beginPath();
      for (var i = 0; i < ts.length; i++) {
        var px = p + (ts[i] - t0) / wd * (w - 2*p);
        var py = h - p - Math.min(a[i], mv) / mv * (h - 2*p);
        if (i === 0) x.moveTo(px, py); else x.lineTo(px, py);
      }
      x.strokeStyle = c; x.lineWidth = lw || 1.5; x.stroke();
    }

    dl(vy, '#ff9800', 2);
    dl(ty, '#e91e63', 1.5);
    dl(vl, '#2196f3', 2);
    dl(tl, '#9c27b0', 1.5);
    dl(fv, '#00bcd4', 1);

    x.fillStyle = '#888'; x.font = '11px sans-serif';
    x.fillText('Velocity (px/s)', 5, 14);
    x.fillText('0', 5, h - 5);
    x.fillText(Math.round(mv) + '', 5, 25);
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
  drawClass(); drawYours(); drawLukas(); drawAgree(); drawVel();
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
                         json.dumps(results))

    output_path = os.path.join(os.path.dirname(__file__), "data",
                               "visualization", "verify_comparison.html")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("=" * 78)
    print("  COMPARISON: Your Code vs Lukas-Wilde Thesis")
    print("=" * 78)
    print(f"\n  Datasets:     {DATASET_ROOT}")
    print(f"  Resolution:   {RESOLUTION}")
    print(f"  Your thresh:  {VELOCITY_THRESHOLD} px/ms ({VELOCITY_THRESHOLD*1000:.0f} px/s)")
    print(f"  Lukas thresh: {LUKAS_THRESHOLD_PXMS} px/ms ({LUKAS_THRESHOLD_PXS:.0f} px/s)")
    print(f"  Savgol:       {SAVGOL_WINDOW_MS} ms")
    print(f"  Gain:         {GAIN}")
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
                r, vis = compare_dataset(category, subject)
                results.append(r)
                vis_data_list.append(vis)
                print(f"  F1 yours={r['f1_yours_fix']:.4f}  "
                      f"lukas={r['f1_lukas_fix']:.4f}  "
                      f"agree={r['agree_flow']:.1f}%  "
                      f"vel_corr={r['vel_correlation']:.4f}")
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    "category": category, "subject": subject,
                    "error": str(e),
                })

    # ── Summary Table ─────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  RESULTS SUMMARY")
    print("=" * 78)

    hdr = (f"{'Dataset':<13} {'Samples':>7} {'Hz':>5} "
           f"{'F1Fx-Y':>8} {'F1Fx-L':>8} {'Delta':>7} "
           f"{'F1Sc-Y':>8} {'F1Sc-L':>8} {'Delta':>7} "
           f"{'F1Fx-Y':>8} {'F1Fx-L':>8} {'Delta':>7} "
           f"{'Agree%':>7} {'VelCorr':>9}")
    sub = (f"{'':13} {'':>7} {'':>5} "
           f"{'(flow)':>8} {'(flow)':>8} {'(fix)':>7} "
           f"{'(flow)':>8} {'(flow)':>8} {'(sac)':>7} "
           f"{'(noFl)':>8} {'(noFl)':>8} {'':>7} "
           f"{'':>7} {'':>9}")
    print(f"\n{hdr}")
    print(sub)
    print("-" * len(hdr))

    ok_results = [r for r in results if "error" not in r]
    for r in results:
        if "error" in r:
            print(f"  {r['category']}/{r['subject']:<8}  ERROR: {r['error']}")
            continue
        sac_delta = r['f1_yours_sac'] - r['f1_lukas_sac']
        print(f"  {r['category']}/{r['subject']:<8} "
              f"{r['n_samples']:>7} {r['sampling_rate']:>5.0f} "
              f"{r['f1_yours_fix']:>8.4f} {r['f1_lukas_fix']:>8.4f} "
              f"{r['f1_delta_flow']:>+7.4f} "
              f"{r['f1_yours_sac']:>8.4f} {r['f1_lukas_sac']:>8.4f} "
              f"{sac_delta:>+7.4f} "
              f"{r['f1_yours_nf_fix']:>8.4f} {r['f1_lukas_nf_fix']:>8.4f} "
              f"{r['f1_delta_noflow']:>+7.4f} "
              f"{r['agree_flow']:>6.1f}% {r['vel_correlation']:>9.6f}")

    if not ok_results:
        print("\n  No datasets processed successfully!")
        return

    # ── Aggregate Statistics ──────────────────────────────────────
    print(f"\n{'─'*78}")
    print("  AGGREGATE STATISTICS")
    print(f"{'─'*78}")

    f1y_flow = [r["f1_yours_fix"] for r in ok_results]
    f1l_flow = [r["f1_lukas_fix"] for r in ok_results]
    f1sy_flow = [r["f1_yours_sac"] for r in ok_results]
    f1sl_flow = [r["f1_lukas_sac"] for r in ok_results]
    f1y_nf = [r["f1_yours_nf_fix"] for r in ok_results]
    f1l_nf = [r["f1_lukas_nf_fix"] for r in ok_results]
    agrees = [r["agree_flow"] for r in ok_results]
    vcorrs = [r["vel_correlation"] for r in ok_results]
    deltas = [r["f1_delta_flow"] for r in ok_results]
    sac_deltas = [r["f1_yours_sac"] - r["f1_lukas_sac"] for r in ok_results]

    print(f"\n  F1-Fix (Yours, flow):   mean={np.mean(f1y_flow):.4f}  "
          f"min={np.min(f1y_flow):.4f}  max={np.max(f1y_flow):.4f}")
    print(f"  F1-Fix (Lukas, flow):   mean={np.mean(f1l_flow):.4f}  "
          f"min={np.min(f1l_flow):.4f}  max={np.max(f1l_flow):.4f}")
    print(f"  F1-Fix Delta (flow):    mean={np.mean(deltas):+.4f}  "
          f"min={np.min(deltas):+.4f}  max={np.max(deltas):+.4f}")
    print(f"  F1-Sac (Yours, flow):   mean={np.mean(f1sy_flow):.4f}  "
          f"min={np.min(f1sy_flow):.4f}  max={np.max(f1sy_flow):.4f}")
    print(f"  F1-Sac (Lukas, flow):   mean={np.mean(f1sl_flow):.4f}  "
          f"min={np.min(f1sl_flow):.4f}  max={np.max(f1sl_flow):.4f}")
    print(f"  F1-Sac Delta (flow):    mean={np.mean(sac_deltas):+.4f}  "
          f"min={np.min(sac_deltas):+.4f}  max={np.max(sac_deltas):+.4f}")
    print(f"  F1-Fix (Yours, no fl):  mean={np.mean(f1y_nf):.4f}")
    print(f"  F1-Fix (Lukas, no fl):  mean={np.mean(f1l_nf):.4f}")
    print(f"  Agreement (flow):       mean={np.mean(agrees):.1f}%  "
          f"min={np.min(agrees):.1f}%")
    print(f"  Velocity correlation:   mean={np.mean(vcorrs):.6f}  "
          f"min={np.min(vcorrs):.6f}")

    # ── Dynamic vs Static comparison ──────────────────────────────
    print(f"\n{'─'*78}")
    print("  DYNAMIC vs STATIC")
    print(f"{'─'*78}")

    for cat in CATEGORIES:
        cr = [r for r in ok_results if r["category"] == cat]
        if not cr:
            continue
        cf_y = [r["f1_yours_fix"] for r in cr]
        cf_l = [r["f1_lukas_fix"] for r in cr]
        cs_y = [r["f1_yours_sac"] for r in cr]
        cs_l = [r["f1_lukas_sac"] for r in cr]
        ca = [r["agree_flow"] for r in cr]
        print(f"\n  {cat.upper()} ({len(cr)} datasets):")
        print(f"    F1-Fix Yours:  mean={np.mean(cf_y):.4f}  "
              f"[{', '.join(f'{v:.4f}' for v in cf_y)}]")
        print(f"    F1-Fix Lukas:  mean={np.mean(cf_l):.4f}  "
              f"[{', '.join(f'{v:.4f}' for v in cf_l)}]")
        print(f"    F1-Sac Yours:  mean={np.mean(cs_y):.4f}  "
              f"[{', '.join(f'{v:.4f}' for v in cs_y)}]")
        print(f"    F1-Sac Lukas:  mean={np.mean(cs_l):.4f}  "
              f"[{', '.join(f'{v:.4f}' for v in cs_l)}]")
        print(f"    Agreement:     mean={np.mean(ca):.1f}%  "
              f"[{', '.join(f'{v:.1f}%' for v in ca)}]")

    # ── Verdict ───────────────────────────────────────────────────
    print(f"\n{'='*78}")
    print("  VERDICT")
    print(f"{'='*78}")

    mean_agree = np.mean(agrees)
    mean_corr = np.mean(vcorrs)

    print(f"\n  Velocity pipelines match:  "
          f"{'PASS' if mean_corr > 0.95 else 'CHECK'}  "
          f"(mean correlation={mean_corr:.4f})")
    print(f"  Classification agreement:  "
          f"{'HIGH' if mean_agree > 90 else ('MODERATE' if mean_agree > 80 else 'LOW')}  "
          f"(mean={mean_agree:.1f}%)")
    print(f"  Note: threshold difference (1000 vs 1200 px/s) "
          f"accounts for most disagreement")

    if mean_corr > 0.95 and mean_agree > 80:
        print(f"\n  >>> OVERALL: Implementations are consistent — "
              f"differences are from threshold choice <<<")
    else:
        print(f"\n  >>> OVERALL: Review differences carefully <<<")

    # ── Generate HTML report ──────────────────────────────────────
    if ok_results and vis_data_list:
        html_path = generate_html(ok_results, vis_data_list)
        print(f"\n  HTML report saved to: {html_path}")
        print(f"  -> Open in browser to see interactive comparison charts")

    print()


if __name__ == "__main__":
    main()
