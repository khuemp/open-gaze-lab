import json
from pathlib import Path

import pandas as pd


_TEMPLATE_PATH = Path(__file__).parent / "_video_template.html"
with open(_TEMPLATE_PATH, "r", encoding="utf-8") as _f:
    _VIDEO_VIS_TEMPLATE = _f.read()


def generate_video_gaze_visualization(
    event_df,
    video_url,
    resolution,
    fps,
    video_start_time=0.0,
    gt_labels_series=None,
    flow_data=None,
    output_path=None,
):
    """Generate a self-contained HTML5 visualization with video + gaze overlay.

    Creates an interactive page with:
      - HTML5 video playback with canvas gaze overlay
      - gaze sample tracking (color-coded fixation/saccade)
      - Trailing gaze path (last 500 ms)
      - Fixation centers with numbered labels and scanpath
      - Optic flow arrow indicating head motion
      - Event timeline strip (clickable for seeking)
      - Play/Pause, speed controls, time slider

    Args:
        event_df: DataFrame from EventDetection with columns:
            x, y, timestamp (ms), event_type, fixation_id, fixation_x, fixation_y
        video_url: URL where the video is served (e.g. /api/video/filename.mp4)
        resolution: Tuple (width, height) of the video
        fps: Video frames per second
        video_start_time: Epoch time (seconds) of the first video frame,
            used to align gaze timestamps to video.currentTime.
        gt_labels_series: Optional Series/array of ground truth labels aligned
            to event_df rows (1=Fixation, 0=Saccade).
        flow_data: Optional list of dicts with per-frame flow info:
            [{time_s, flow_x, flow_y}, ...]
        output_path: If provided, write HTML to this file path.

    Returns:
        The HTML content string.
    """
    res_w, res_h = resolution
    df = event_df.copy()

    # Convert timestamps from ms back to seconds relative to video start
    gaze_time_s = df["timestamp"].values / 1000.0  # ms -> absolute seconds
    # The EventDetection constructor converted from seconds->ms.  The original
    # .npy timestamps shared the same epoch as time_scene_camera.  Subtract
    # video_start_time so that t=0 aligns with <video>.currentTime=0.
    gaze_time_rel = gaze_time_s - video_start_time

    # Downsample for reasonable HTML size (target ~3000 points)
    n = len(df)
    step = max(1, n // 3000)

    # Pre-align GT to df.index so positional access matches df.iloc rows.
    # gt_labels_series carries the *original* gaze_df index, while df may be
    # a filtered subset (e.g. NaN/out-of-range rows removed) — using iloc
    # directly on the unaligned series would produce off-by-N mismatches.
    gt_aligned_arr = None
    if gt_labels_series is not None:
        gt_aligned_arr = gt_labels_series.loc[df.index].values.astype(int)

    gaze_samples = []
    for i in range(0, n, step):
        row = df.iloc[i]
        sample = {
            "t": round(float(gaze_time_rel[i]), 4),
            "x": round(float(row["x"]), 1),
            "y": round(float(row["y"]), 1),
            "ev": 1 if row["event_type"] == "Fixation" else 0,
            "fid": int(row["fixation_id"]) if pd.notna(row.get("fixation_id")) else -1,
        }
        if gt_aligned_arr is not None:
            sample["gt"] = int(gt_aligned_arr[i])
        gaze_samples.append(sample)

    # Build fixation summaries
    fix_groups = df[df["event_type"] == "Fixation"].groupby("fixation_id")
    fixation_summaries = []
    for fid, grp in fix_groups:
        t_start = float(grp["timestamp"].iloc[0]) / 1000.0 - video_start_time
        t_end = float(grp["timestamp"].iloc[-1]) / 1000.0 - video_start_time
        fixation_summaries.append({
            "id": int(fid),
            "cx": round(float(grp["fixation_x"].iloc[0]), 1),
            "cy": round(float(grp["fixation_y"].iloc[0]), 1),
            "ts": round(t_start, 4),
            "te": round(t_end, 4),
            "dur": round(t_end - t_start, 4),
        })
    fixation_summaries.sort(key=lambda f: f["ts"])

    # Flow data for arrow rendering
    flow_json = json.dumps(flow_data if flow_data else [])

    # Stats
    n_fixations = df["fixation_id"].dropna().nunique()
    n_saccade_points = len(df[df["event_type"] == "Saccade"])
    n_fixation_points = len(df[df["event_type"] == "Fixation"])
    duration_s = gaze_time_rel[-1] - gaze_time_rel[0] if len(gaze_time_rel) > 1 else 0

    stats = {
        "n_fixations": int(n_fixations),
        "n_fixation_points": int(n_fixation_points),
        "n_saccade_points": int(n_saccade_points),
        "duration_s": round(float(duration_s), 2),
    }

    # GT comparison stats (reuse the index-aligned array built above)
    if gt_aligned_arr is not None:
        from sklearn.metrics import f1_score
        pred = (df["event_type"] == "Fixation").astype(int).values
        stats["f1_fixation"] = round(float(f1_score(gt_aligned_arr, pred, pos_label=1, zero_division=0)), 4)
        stats["f1_saccade"] = round(float(f1_score(gt_aligned_arr, pred, pos_label=0, zero_division=0)), 4)

    has_gt = gt_labels_series is not None

    html = _VIDEO_VIS_TEMPLATE.replace("__GAZE_DATA__", json.dumps(gaze_samples))
    html = html.replace("__FIXATIONS__", json.dumps(fixation_summaries))
    html = html.replace("__FLOW_DATA__", flow_json)
    html = html.replace("__STATS__", json.dumps(stats))
    html = html.replace("__VIDEO_URL__", video_url)
    html = html.replace("__RES_W__", str(res_w))
    html = html.replace("__RES_H__", str(res_h))
    html = html.replace("__FPS__", str(round(fps, 2)))
    html = html.replace("__HAS_GT__", "true" if has_gt else "false")

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

    return html
