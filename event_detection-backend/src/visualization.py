
import json
import base64
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as pyo


def _encode_image_base64(image_path):
    """Encode an image file to a base64 data URI for Plotly."""
    ext = image_path.lower().split('.')[-1]
    mime_types = {'png': 'image/png', 'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 
                  'gif': 'image/gif', 'webp': 'image/webp', 'bmp': 'image/bmp'}
    mime_type = mime_types.get(ext, 'image/png')
    
    with open(image_path, 'rb') as f:
        encoded = base64.b64encode(f.read()).decode('utf-8')
    return f"data:{mime_type};base64,{encoded}"


def plot_gaze_points_and_fixations(self, output_dir, bg_image_path=None, aois=None, show_attach=True,
                                    attach_type='bbox', y_origin='top-left'):
    """Creates an interactive visualization of gaze data, fixations, and AOIs.

    Generates a Plotly-based interactive plot showing gaze points, fixations, and
    optionally Areas of Interest (AOIs). The plot can include a background image
    and different styles of AOI visualization.

    Args:
        output_dir (str): Path where the HTML plot file will be saved
        bg_image_path (str, optional): Path to background image file. Defaults to None.
        aois (pd.DataFrame, optional): AOI definitions with columns:
            - aoi_type (str): Category of the AOI
            - aoi (str): Name of the AOI
            - pos_x (float): X coordinate of top-left corner
            - pos_y (float): Y coordinate of top-left corner
            - width (float): Width of the AOI
            - height (float): Height of the AOI
            Defaults to None.
        show_attach (bool, optional): Whether to show AOI attachments. Defaults to True.
        attach_type (str, optional): Style of AOI visualization:
            - 'centroid': Show center points
            - 'bbox': Show bounding boxes
            Defaults to 'bbox'.
        y_origin (str, optional): Origin position for the coordinate system.
            One of 'top-left', 'top-right', 'bottom-left', 'bottom-right'.
            Defaults to 'top-left'.

    Returns:
        None: Saves an interactive HTML plot to the specified output directory
    
    Notes:
        Requires self.event_data_df to contain:
            - x, y: Raw gaze coordinates
            - fixation_x, fixation_y: Fixation center coordinates
            - event_type: 'Fixation' or 'Saccade'
    """

    gaze_data = self.event_data_df.copy()
    res_w, res_h = self.resolution

    # Determine axis ranges based on origin
    flip_x = y_origin in ('top-right', 'bottom-right')
    flip_y = y_origin in ('top-left', 'top-right')  # screen coords: y=0 at top
    x_range = [res_w, 0] if flip_x else [0, res_w]
    y_range = [res_h, 0] if flip_y else [0, res_h]

    # Color palette
    color_map = {
        'Fixation': 'rgba(34, 139, 34, 0.6)',
        'Saccade': 'rgba(100, 100, 100, 0.4)',
    }
    colors = [color_map.get(e, color_map['Saccade']) for e in gaze_data['event_type']]

    # Gaze points scatter
    gaze_scatter = go.Scatter(
        x=gaze_data['x'],
        y=gaze_data['y'],
        mode='markers',
        marker=dict(color=colors, size=6, opacity=0.7),
        name='Gaze Points'
    )

    # Fixation centers
    fixations = gaze_data[['fixation_x', 'fixation_y', 'fixation_id']].drop_duplicates(subset=['fixation_id'])
    fixations = fixations[fixations['fixation_x'].notna() & fixations['fixation_y'].notna()]
    fixations = fixations.sort_values('fixation_id').reset_index(drop=True)

    fixation_scatter = go.Scatter(
        x=fixations['fixation_x'],
        y=fixations['fixation_y'],
        mode='markers+text',
        marker=dict(color='#1a1a1a', size=12, line=dict(color='white', width=2)),
        text=[str(int(fid)) for fid in fixations['fixation_id']],
        textposition='top center',
        textfont=dict(size=10, color='#1a1a1a', family='Arial'),
        name='Fixation Centers'
    )

    # Scanpath
    fixation_line = go.Scatter(
        x=fixations['fixation_x'],
        y=fixations['fixation_y'],
        mode='lines',
        line=dict(color='rgba(26, 26, 26, 0.5)', width=1.5),
        name='Scanpath'
    )

    # Layout - hide grid when background image is present
    show_grid = bg_image_path is None
    plot_bg = '#fafafa' if bg_image_path is None else 'rgba(0,0,0,0)'
    
    layout = go.Layout(
        title=dict(
            text='Gaze and Fixation Visualization',
            font=dict(size=16, color='#1a1a1a', family='Arial, sans-serif'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title=dict(text='X Position (px)', font=dict(size=11, color='#666')),
            range=x_range,
            autorange=False,
            showgrid=show_grid,
            gridcolor='rgba(200, 200, 200, 0.3)',
            zeroline=False,
            showline=True,
            linecolor='#ddd',
            tickfont=dict(size=10, color='#666'),
            scaleanchor='y',
            scaleratio=1,
            constrain='domain'
        ),
        yaxis=dict(
            title=dict(text='Y Position (px)', font=dict(size=11, color='#666')),
            range=y_range,
            autorange=False,
            showgrid=show_grid,
            gridcolor='rgba(200, 200, 200, 0.3)',
            zeroline=False,
            showline=True,
            linecolor='#ddd',
            tickfont=dict(size=10, color='#666'),
            constrain='domain'
        ),
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            font=dict(size=10, color='#333'),
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='#ddd',
            borderwidth=1
        ),
        plot_bgcolor=plot_bg,
        paper_bgcolor='white',
        margin=dict(l=60, r=40, t=80, b=60)
    )

    # Combine visualization layers
    data = [gaze_scatter, fixation_scatter, fixation_line]
    fig = go.Figure(data=data, layout=layout)

    if bg_image_path is not None:
        try:
            image_source = _encode_image_base64(bg_image_path)
            # Position image at the visual top-left corner of the plot.
            # Plotly layout images always extend rightward and downward in
            # pixel space from the anchor, so we always use xanchor='left'
            # and yanchor='top'. The data coordinates for the top-left pixel
            # depend on axis direction.
            img_x = res_w if flip_x else 0
            img_y = 0 if flip_y else res_h
            fig.add_layout_image(dict(
                source=image_source,
                xref="x", yref="y",
                x=img_x, y=img_y,
                sizex=res_w, sizey=res_h,
                xanchor='left', yanchor='top',
                sizing="stretch",
                opacity=0.35,
                layer="below"
            ))
        except Exception as e:
            print(f"Warning: Could not load background image: {e}")

    if aois is not None:
        for index, aoi in aois.iterrows():
            fig.add_shape(
                type="rect",
                xref="x", yref="y",
                x0=aoi['pos_x'],
                y0=aoi['pos_y'],
                x1=aoi['pos_x'] + aoi['width'],
                y1=aoi['pos_y'] + aoi['height'],
                line=dict(color='#5a7a8a', width=2),
                fillcolor='rgba(90, 122, 138, 0.12)',
                layer="below"
            )
            fig.add_annotation(
                x=aoi['pos_x'] + aoi['width'] / 2,
                y=aoi['pos_y'] + aoi['height'] / 2,
                text=aoi['aoi'],
                showarrow=False,
                font=dict(color='#5a7a8a', size=11, family='Arial')
            )

    if show_attach and 'aoi' in gaze_data.columns and aois is not None:
        fixations = gaze_data[gaze_data['aoi_type'].notna()][
            ['fixation_x', 'fixation_y', 'aoi_type', 'aoi', 'aoi_id']].drop_duplicates()
        fixations.reset_index(drop=True, inplace=True)

        for idx, fixation in fixations.iterrows():
            if fixation['aoi_type'] == 'word':
                aoi = aois.iloc[int(fixation['aoi_id'])]
                aoi_center_x = aoi['pos_x'] + aoi['width'] / 2
                aoi_center_y = aoi['pos_y'] + aoi['height'] / 2
            else:
                continue

            fig.add_trace(go.Scatter(
                x=[fixation['fixation_x'], aoi_center_x],
                y=[fixation['fixation_y'], aoi_center_y],
                mode='lines',
                line=dict(color='rgba(160, 90, 90, 0.5)', width=1.5, dash='dot'),
                showlegend=False,
                hoverinfo='skip'
            ))

    # Save the figure to an HTML file
    plot_config = {
        'toImageButtonOptions': {
            'format': 'png',
            'width': res_w,
            'height': res_h,
            'scale': 3
        }
    }
    pyo.plot(fig, filename=output_dir, auto_open=False, config=plot_config)


def plot_gaze_with_time_scrolling(self, output_dir, bg_image_path=None, aois=None, 
                                   time_window_ms=5000, step_ms=100, y_origin='top-left'):
    """Creates an interactive time-scrollable visualization of gaze data and fixations.

    Generates a Plotly-based animated plot that allows users to scroll through time
    and see how gaze points and fixations appear progressively. Includes a time slider
    and play/pause controls for temporal exploration of eye movement data.

    Args:
        output_dir (str): Path where the HTML plot file will be saved
        bg_image_path (str, optional): Path to background image file. Defaults to None.
        aois (pd.DataFrame, optional): AOI definitions with columns:
            - aoi_type (str): Category of the AOI
            - aoi (str): Name of the AOI
            - pos_x (float): X coordinate of top-left corner
            - pos_y (float): Y coordinate of top-left corner
            - width (float): Width of the AOI
            - height (float): Height of the AOI
            Defaults to None.
        time_window_ms (float, optional): Duration of the visible time window in milliseconds.
            Defaults to 5000ms (5 seconds).
        step_ms (float, optional): Time step between animation frames in milliseconds.
            Defaults to 100ms.
        y_origin (str, optional): Origin position for the coordinate system.
            One of 'top-left', 'top-right', 'bottom-left', 'bottom-right'.
            Defaults to 'top-left'.

    Returns:
        None: Saves an interactive HTML plot with time controls to the specified directory
    
    Notes:
        Requires self.event_data_df to contain:
            - x, y: Raw gaze coordinates
            - fixation_x, fixation_y: Fixation center coordinates
            - event_type: 'Fixation' or 'Saccade'
            - timestamp: Time in milliseconds
    """
    
    gaze_data = self.event_data_df.copy()
    res_w, res_h = self.resolution

    # Determine axis ranges based on origin
    flip_x = y_origin in ('top-right', 'bottom-right')
    flip_y = y_origin in ('top-left', 'top-right')  # screen coords: y=0 at top
    x_range = [res_w, 0] if flip_x else [0, res_w]
    y_range = [res_h, 0] if flip_y else [0, res_h]
    
    # Sort by timestamp to ensure correct temporal sequence
    gaze_data = gaze_data.sort_values('timestamp').reset_index(drop=True)
    
    # Professional minimalist color palette
    color_map = {'Fixation': 'rgba(34, 139, 34, 0.6)',    # Forest green for fixations
                 'Saccade': 'rgba(70, 70, 70, 0.4)'}       # Dark gray for saccades
    
    # Get time range
    min_time = gaze_data['timestamp'].min()
    max_time = gaze_data['timestamp'].max()
    time_steps = list(range(int(min_time), int(max_time) + int(step_ms), int(step_ms)))
    
    # Build animation frames
    frames = []
    for t in time_steps:
        visible = gaze_data[gaze_data['timestamp'] <= t]
        if len(visible) == 0:
            continue
        
        colors = [color_map.get(e, color_map['Saccade']) for e in visible['event_type']]
        
        # Use original index as ids to maintain point identity across frames
        gaze_scatter = go.Scatter(
            x=visible['x'].tolist(),
            y=visible['y'].tolist(),
            ids=visible.index.astype(str).tolist(),
            mode='markers',
            marker=dict(color=colors, size=6, opacity=0.7),
            name='Gaze Points',
            showlegend=True
        )
        
        fixations = visible[['fixation_x', 'fixation_y', 'fixation_id']].drop_duplicates(subset=['fixation_id'])
        fixations = fixations[fixations['fixation_x'].notna() & fixations['fixation_y'].notna()]
        fixations = fixations.sort_values('fixation_id')
        
        fixation_scatter = go.Scatter(
            x=fixations['fixation_x'].tolist(),
            y=fixations['fixation_y'].tolist(),
            ids=fixations['fixation_id'].astype(str).tolist(),
            mode='markers+text',
            marker=dict(color='#1a1a1a', size=12, line=dict(color='white', width=2)),
            text=[str(int(fid)) for fid in fixations['fixation_id']],
            textposition='top center',
            textfont=dict(size=10, color='#1a1a1a', family='Arial'),
            name='Fixation Centers',
            showlegend=True
        )
        
        fixation_line = go.Scatter(
            x=fixations['fixation_x'].tolist(),
            y=fixations['fixation_y'].tolist(),
            mode='lines',
            line=dict(color='rgba(26, 26, 26, 0.5)', width=1.5),
            name='Scanpath',
            showlegend=True
        )
        
        frames.append(go.Frame(
            data=[gaze_scatter, fixation_scatter, fixation_line],
            name=str(t)
        ))
    
    # Initial frame
    initial_data = frames[0].data if frames else []
    
    # Layout - hide grid when background image is present
    show_grid = bg_image_path is None
    plot_bg = '#fafafa' if bg_image_path is None else 'rgba(0,0,0,0)'
    
    layout = go.Layout(
        title=dict(
            text='Gaze and Fixation Time-Scrolling Visualization',
            font=dict(size=16, color='#1a1a1a', family='Arial, sans-serif'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title=dict(text='X Position (px)', font=dict(size=11, color='#666')),
            range=x_range,
            autorange=False,
            showgrid=show_grid,
            gridcolor='rgba(200, 200, 200, 0.3)',
            zeroline=False,
            showline=True,
            linecolor='#ddd',
            tickfont=dict(size=10, color='#666'),
            scaleanchor='y',
            scaleratio=1,
            constrain='domain'
        ),
        yaxis=dict(
            title=dict(text='Y Position (px)', font=dict(size=11, color='#666')),
            range=y_range,
            autorange=False,
            showgrid=show_grid,
            gridcolor='rgba(200, 200, 200, 0.3)',
            zeroline=False,
            showline=True,
            linecolor='#ddd',
            tickfont=dict(size=10, color='#666'),
            constrain='domain'
        ),
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            font=dict(size=10, color='#333'),
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='#ddd',
            borderwidth=1
        ),
        plot_bgcolor=plot_bg,
        paper_bgcolor='white',
        margin=dict(l=60, r=40, t=80, b=100),
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'bgcolor': '#f0f0f0',
            'bordercolor': '#ccc',
            'font': dict(size=11, color='#333'),
            'buttons': [
                {
                    'label': 'Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 300, 'redraw': True},
                        'fromcurrent': True,
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                },
                {
                    'label': 'Pause',
                    'method': 'animate',
                    'args': [[None], {
                        'frame': {'duration': 0, 'redraw': False},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                }
            ],
            'x': 0.0,
            'y': -0.12,
            'xanchor': 'left',
            'yanchor': 'top'
        }],
        sliders=[{
            'active': 0,
            'bgcolor': '#e0e0e0',
            'bordercolor': '#ccc',
            'tickcolor': '#999',
            'font': dict(size=9, color='#666'),
            'steps': [
                {
                    'label': '',
                    'method': 'animate',
                    'args': [[str(t)], {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                }
                for t in time_steps
            ],
            'x': 0.12,
            'y': -0.06,
            'len': 0.88,
            'xanchor': 'left',
            'yanchor': 'top',
            'pad': {'b': 10, 't': 20},
            'currentvalue': {
                'visible': True,
                'prefix': 'Time: ',
                'suffix': ' ms',
                'xanchor': 'left',
                'font': dict(size=11, color='#333')
            },
            'transition': {'duration': 0}
        }]
    )
    
    # Create figure with frames
    fig = go.Figure(data=initial_data, layout=layout, frames=frames)
    
    if bg_image_path is not None:
        try:
            image_source = _encode_image_base64(bg_image_path)
            # Position image at the visual top-left corner of the plot.
            # Plotly layout images always extend rightward and downward in
            # pixel space from the anchor, so we always use xanchor='left'
            # and yanchor='top'. The data coordinates for the top-left pixel
            # depend on axis direction.
            img_x = res_w if flip_x else 0
            img_y = 0 if flip_y else res_h
            fig.add_layout_image(dict(
                source=image_source,
                xref="x", yref="y",
                x=img_x, y=img_y,
                sizex=res_w, sizey=res_h,
                xanchor='left', yanchor='top',
                sizing="stretch",
                opacity=0.35,
                layer="below"
            ))
        except Exception as e:
            print(f"Warning: Could not load background image: {e}")
    
    if aois is not None:
        for index, aoi in aois.iterrows():
            fig.add_shape(
                type="rect",
                xref="x", yref="y",
                x0=aoi['pos_x'],
                y0=aoi['pos_y'],
                x1=aoi['pos_x'] + aoi['width'],
                y1=aoi['pos_y'] + aoi['height'],
                line=dict(color='#5a7a8a', width=2),
                fillcolor='rgba(90, 122, 138, 0.12)',
                layer="below"
            )
            fig.add_annotation(
                x=aoi['pos_x'] + aoi['width'] / 2,
                y=aoi['pos_y'] + aoi['height'] / 2,
                text=aoi['aoi'],
                showarrow=False,
                font=dict(color='#5a7a8a', size=11, family='Arial')
            )
    
    plot_config = {
        'toImageButtonOptions': {
            'format': 'png',
            'width': res_w,
            'height': res_h,
            'scale': 3
        }
    }
    pyo.plot(fig, filename=output_dir, auto_open=False, config=plot_config)


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
      - Gaze point tracking (color-coded fixation/saccade)
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
        if gt_labels_series is not None:
            sample["gt"] = int(gt_labels_series.iloc[i])
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

    # GT comparison stats
    if gt_labels_series is not None:
        from sklearn.metrics import f1_score
        pred = (df["event_type"] == "Fixation").astype(int).values
        gt = gt_labels_series.values.astype(int)
        stats["f1_fixation"] = round(float(f1_score(gt, pred, pos_label=1, zero_division=0)), 4)
        stats["f1_saccade"] = round(float(f1_score(gt, pred, pos_label=0, zero_division=0)), 4)

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


# ---------------------------------------------------------------------------
# HTML template for the video gaze visualization
# ---------------------------------------------------------------------------

_VIDEO_VIS_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Head-Mounted Eye Tracking Visualization</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: #1a1a2e; color: #e0e0e0; padding: 10px;
  }
  h1 { display: none; }
  .subtitle { display: none; }

  /* Video + canvas wrapper */
  .main-layout {
    display: flex; gap: 12px; margin-bottom: 12px; align-items: flex-start;
  }
  .video-wrap {
    position: relative; display: inline-block;
    background: #000; border-radius: 8px; overflow: hidden;
    flex: 1; min-width: 0;
  }
  .video-wrap video {
    display: block; width: 100%; height: auto;
  }
  .video-wrap canvas {
    position: absolute; top: 0; left: 0;
    width: 100%; height: 100%;
    pointer-events: none;
  }
  .sidebar {
    width: 170px; flex-shrink: 0;
    display: flex; flex-direction: column; gap: 10px;
  }
  .sidebar .legend { display: flex; flex-direction: column; gap: 6px; font-size: 0.85em; margin: 0; }
  .sidebar .legend span { display: flex; align-items: center; gap: 5px; }
  .sidebar .legend .dot { width: 11px; height: 11px; border-radius: 50%; display: inline-block; }
  .sidebar .toggle-row { display: flex; flex-direction: column; gap: 6px; margin: 0; }
  .sidebar .toggle-row label {
    display: flex; align-items: center; gap: 5px;
    background: #16213e; padding: 5px 10px; border-radius: 6px;
    font-size: 0.82em; cursor: pointer; user-select: none;
    border: 1px solid #333;
  }
  .sidebar .toggle-row input[type=checkbox] { accent-color: #6dd5fa; }

  /* Timeline strip */
  .timeline-wrap {
    background: #16213e; border-radius: 6px; padding: 8px 12px;
    margin-bottom: 10px; cursor: pointer;
  }
  .timeline-wrap canvas { width: 100%; height: 36px; border-radius: 4px; }

  /* Controls row */
  .controls {
    display: flex; align-items: center; gap: 12px;
    margin-bottom: 14px; flex-wrap: wrap;
  }
  .controls button {
    background: #16213e; color: #6dd5fa; border: 1px solid #6dd5fa;
    padding: 6px 16px; border-radius: 6px; cursor: pointer;
    font-size: 0.9em; transition: background .2s;
  }
  .controls button:hover { background: #6dd5fa22; }
  .controls button.active { background: #6dd5fa; color: #1a1a2e; }
  .time-display { color: #6dd5fa; font-variant-numeric: tabular-nums;
                  font-size: 0.95em; min-width: 120px; }
  .speed-group { display: flex; gap: 4px; }
  .speed-group button { padding: 4px 10px; font-size: 0.82em; }
  input[type=range] {
    flex: 1; min-width: 200px; accent-color: #6dd5fa;
  }

  /* Toggle buttons for overlay layers - handled in sidebar */
  .toggle-row { display: none; }
  .toggle-row input[type=checkbox] { accent-color: #6dd5fa; }

  /* Stats cards - hidden, shown in parent page */
  .stats { display: none; }

  /* Legend - handled in sidebar */
  .legend { display: none; }
  .legend span { display: flex; align-items: center; gap: 5px; }
  .legend .dot { width: 11px; height: 11px; border-radius: 50%; display: inline-block; }
</style>
</head>
<body>
<h1>Head-Mounted Eye Tracking Visualization</h1>
<p class="subtitle">Gaze overlay on scene camera video &mdash; I-VAT+Frel pipeline</p>

<!-- Main layout: video + sidebar -->
<div class="main-layout">

<!-- Video with canvas overlay -->
<div class="video-wrap" id="videoWrap">
  <video id="vid" preload="auto" muted>
    <source src="__VIDEO_URL__" type="video/mp4">
  </video>
  <canvas id="overlay"></canvas>
</div>

<!-- Sidebar: legend + toggles -->
<div class="sidebar">
  <div class="legend">
    <span><span class="dot" style="background:#22c55e"></span> Fixation</span>
    <span><span class="dot" style="background:#ef4444"></span> Saccade</span>
    <span><span class="dot" style="background:#6dd5fa"></span> Optic Flow</span>
    <span id="gt-legend-side" style="display:none"><span class="dot" style="background:#fbbf24"></span> Ground Truth</span>
  </div>
  <div class="toggle-row" style="display:flex">
    <label><input type="checkbox" id="tGaze" checked> Gaze Point</label>
    <label><input type="checkbox" id="tTrail" checked> Gaze Trail</label>
    <label><input type="checkbox" id="tFix" checked> Fixation Centers</label>
    <label><input type="checkbox" id="tScan" checked> Scanpath</label>
    <label><input type="checkbox" id="tFlow" checked> Flow Arrow</label>
    <label id="tGtLabel" style="display:none"><input type="checkbox" id="tGt" checked> GT Comparison</label>
  </div>
</div>

</div><!-- end main-layout -->

<!-- Timeline strip -->
<div class="timeline-wrap" id="timelineWrap">
  <canvas id="timeline" height="36"></canvas>
</div>

<!-- Controls -->
<div class="controls">
  <button id="btnPlay">&#9654; Play</button>
  <button id="btnPause">&#9646;&#9646; Pause</button>
  <div class="speed-group">
    <button data-speed="0.5">0.5&times;</button>
    <button data-speed="1" class="active">1&times;</button>
    <button data-speed="2">2&times;</button>
  </div>
  <input type="range" id="seekBar" min="0" max="1000" value="0" step="1">
  <span class="time-display" id="timeDisp">0.00 / 0.00 s</span>
</div>

<!-- Stats cards -->
<div class="stats" id="statsArea"></div>

<script>
(function(){
  "use strict";

  /* -- Embedded data -- */
  var GAZE   = __GAZE_DATA__;
  var FIX    = __FIXATIONS__;
  var FLOW   = __FLOW_DATA__;
  var STATS  = __STATS__;
  var RES_W  = __RES_W__;
  var RES_H  = __RES_H__;
  var FPS    = __FPS__;
  var HAS_GT = __HAS_GT__;

  /* -- DOM refs -- */
  var vid      = document.getElementById("vid");
  var overlay  = document.getElementById("overlay");
  var ctx      = overlay.getContext("2d");
  var tlCanvas = document.getElementById("timeline");
  var tlCtx    = tlCanvas.getContext("2d");
  var seekBar  = document.getElementById("seekBar");
  var timeDisp = document.getElementById("timeDisp");
  var btnPlay  = document.getElementById("btnPlay");
  var btnPause = document.getElementById("btnPause");
  var statsArea= document.getElementById("statsArea");

  /* Toggle checkboxes */
  var tGaze = document.getElementById("tGaze");
  var tTrail= document.getElementById("tTrail");
  var tFix  = document.getElementById("tFix");
  var tScan = document.getElementById("tScan");
  var tFlow = document.getElementById("tFlow");
  var tGt   = document.getElementById("tGt");

  if(HAS_GT){
    document.getElementById("gt-legend-side").style.display="";
    document.getElementById("tGtLabel").style.display="";
  }

  /* -- Sizing -- */
  function resize(){
    overlay.width  = vid.videoWidth  || RES_W;
    overlay.height = vid.videoHeight || RES_H;
    var tlW = document.getElementById("timelineWrap").clientWidth - 24;
    tlCanvas.width  = tlW > 0 ? tlW : 600;
    tlCanvas.height = 36;
    drawTimeline();
  }
  vid.addEventListener("loadedmetadata", resize);
  window.addEventListener("resize", resize);

  /* -- Binary search helper -- */
  function findIndex(arr, t){
    var lo=0, hi=arr.length-1;
    while(lo<=hi){
      var mid=(lo+hi)>>>1;
      if(arr[mid].t<t) lo=mid+1; else hi=mid-1;
    }
    return lo;
  }

  /* -- Flow binary search -- */
  function findFlowAtTime(t){
    if(!FLOW.length) return null;
    var lo=0, hi=FLOW.length-1;
    while(lo<=hi){
      var mid=(lo+hi)>>>1;
      if(FLOW[mid].time_s<t) lo=mid+1; else hi=mid-1;
    }
    var idx = Math.min(Math.max(lo-1,0), FLOW.length-1);
    return FLOW[idx];
  }

  /* -- Draw timeline strip (static) -- */
  function drawTimeline(){
    var w=tlCanvas.width, h=tlCanvas.height;
    tlCtx.clearRect(0,0,w,h);
    if(!GAZE.length) return;
    var t0=GAZE[0].t, t1=GAZE[GAZE.length-1].t, dur=t1-t0||1;
    for(var i=0;i<GAZE.length-1;i++){
      var x0=(GAZE[i].t-t0)/dur*w;
      var x1=(GAZE[i+1].t-t0)/dur*w;
      tlCtx.fillStyle = GAZE[i].ev ? "#22c55e" : "#ef4444";
      tlCtx.fillRect(x0,0,Math.max(x1-x0,1),h);
    }
    /* GT row if available */
    if(HAS_GT && tGt && tGt.checked){
      var gh=8;
      for(var i=0;i<GAZE.length-1;i++){
        var x0=(GAZE[i].t-t0)/dur*w;
        var x1=(GAZE[i+1].t-t0)/dur*w;
        tlCtx.fillStyle = GAZE[i].gt===1 ? "rgba(251,191,36,0.8)" : "rgba(168,85,247,0.8)";
        tlCtx.fillRect(x0,h-gh,Math.max(x1-x0,1),gh);
      }
    }
  }

  /* -- Draw playhead on timeline -- */
  function drawPlayhead(currentT){
    drawTimeline();
    if(!GAZE.length) return;
    var w=tlCanvas.width, h=tlCanvas.height;
    var t0=GAZE[0].t, t1=GAZE[GAZE.length-1].t, dur=t1-t0||1;
    var x=(currentT-t0)/dur*w;
    tlCtx.save();
    tlCtx.strokeStyle="#fff";
    tlCtx.lineWidth=2;
    tlCtx.beginPath(); tlCtx.moveTo(x,0); tlCtx.lineTo(x,h); tlCtx.stroke();
    /* triangle marker */
    tlCtx.fillStyle="#fff";
    tlCtx.beginPath(); tlCtx.moveTo(x-5,0); tlCtx.lineTo(x+5,0); tlCtx.lineTo(x,7); tlCtx.fill();
    tlCtx.restore();
  }

  /* -- Timeline click to seek -- */
  document.getElementById("timelineWrap").addEventListener("click", function(e){
    if(!GAZE.length) return;
    var rect=tlCanvas.getBoundingClientRect();
    var frac=(e.clientX-rect.left)/rect.width;
    var t0=GAZE[0].t, t1=GAZE[GAZE.length-1].t;
    vid.currentTime = t0 + frac*(t1-t0);
  });

  /* -- Main overlay draw -- */
  function drawOverlay(){
    var cw=overlay.width, ch=overlay.height;
    ctx.clearRect(0,0,cw,ch);
    var currentT = vid.currentTime;
    var scaleX = cw/RES_W, scaleY = ch/RES_H;

    /* Find gaze samples near current time */
    var idx = findIndex(GAZE, currentT);
    var trailDur = 0.5; /* seconds of trail */

    /* -- Gaze trail -- */
    if(tTrail.checked){
      ctx.save();
      for(var i=idx-1; i>=0; i--){
        var age = currentT - GAZE[i].t;
        if(age > trailDur) break;
        var alpha = 1.0 - age/trailDur;
        var g = GAZE[i];
        ctx.fillStyle = g.ev
          ? "rgba(34,197,94,"+(alpha*0.6).toFixed(2)+")"
          : "rgba(239,68,68,"+(alpha*0.4).toFixed(2)+")";
        ctx.beginPath();
        ctx.arc(g.x*scaleX, g.y*scaleY, 4, 0, Math.PI*2);
        ctx.fill();
      }
      ctx.restore();
    }

    /* -- Current gaze point -- */
    if(tGaze.checked && idx > 0 && idx <= GAZE.length){
      var g = GAZE[Math.min(idx, GAZE.length-1)];
      /* only show if within 0.1 s of a sample */
      if(Math.abs(g.t - currentT) < 0.1){
        var color = g.ev ? "#22c55e" : "#ef4444";
        var px = g.x*scaleX, py = g.y*scaleY;
        /* outer glow */
        ctx.save();
        ctx.shadowColor = color;
        ctx.shadowBlur = 14;
        ctx.fillStyle = color;
        ctx.beginPath(); ctx.arc(px,py,8,0,Math.PI*2); ctx.fill();
        ctx.restore();
        /* inner dot */
        ctx.fillStyle = "#fff";
        ctx.beginPath(); ctx.arc(px,py,3,0,Math.PI*2); ctx.fill();
      }
    }

    /* -- Fixation centers + scanpath visible up to current time (last 5 only) -- */
    var visibleFix = [];
    for(var fi=0;fi<FIX.length;fi++){
      if(FIX[fi].ts <= currentT) visibleFix.push(FIX[fi]);
    }
    var recentFix = visibleFix.slice(-5);
    if(tScan.checked && recentFix.length > 1){
      ctx.save();
      ctx.strokeStyle = "rgba(255,255,255,0.35)";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([6,4]);
      ctx.beginPath();
      ctx.moveTo(recentFix[0].cx*scaleX, recentFix[0].cy*scaleY);
      for(var i=1;i<recentFix.length;i++){
        ctx.lineTo(recentFix[i].cx*scaleX, recentFix[i].cy*scaleY);
      }
      ctx.stroke();
      ctx.restore();
    }

    if(tFix.checked){
      for(var fi=0;fi<recentFix.length;fi++){
        var f=recentFix[fi];
        var px=f.cx*scaleX, py=f.cy*scaleY;
        var r = Math.min(6 + f.dur*30, 20); /* radius proportional to dwell time */
        var isActive = currentT >= f.ts && currentT <= f.te;
        /* circle */
        ctx.save();
        ctx.fillStyle = isActive ? "rgba(34,197,94,0.5)" : "rgba(255,255,255,0.15)";
        ctx.strokeStyle = isActive ? "#22c55e" : "rgba(255,255,255,0.5)";
        ctx.lineWidth = isActive ? 2.5 : 1.5;
        ctx.beginPath(); ctx.arc(px,py,r,0,Math.PI*2);
        ctx.fill(); ctx.stroke();
        ctx.restore();
        /* label */
        ctx.save();
        ctx.fillStyle = "#fff";
        ctx.font = "bold 11px 'Segoe UI',sans-serif";
        ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.fillText(String(f.id), px, py);
        ctx.restore();
      }
    }

    /* -- Optic flow arrow -- */
    if(tFlow.checked){
      var fl = findFlowAtTime(currentT);
      if(fl){
        var mag = Math.sqrt(fl.flow_x*fl.flow_x + fl.flow_y*fl.flow_y);
        var maxLen = 60;
        var len = Math.min(mag * 12, maxLen);
        if(len > 2){
          var ax = cw - 60, ay = 50; /* top-right corner anchor */
          var angle = Math.atan2(fl.flow_y, fl.flow_x);
          var ex = ax + Math.cos(angle)*len;
          var ey = ay + Math.sin(angle)*len;
          var alpha = Math.min(0.4 + mag*0.3, 1.0);
          ctx.save();
          ctx.strokeStyle = "rgba(109,213,250,"+alpha.toFixed(2)+")";
          ctx.fillStyle   = "rgba(109,213,250,"+alpha.toFixed(2)+")";
          ctx.lineWidth = 2.5;
          ctx.beginPath(); ctx.moveTo(ax,ay); ctx.lineTo(ex,ey); ctx.stroke();
          /* arrowhead */
          var headLen=8, a1=angle+2.7, a2=angle-2.7;
          ctx.beginPath();
          ctx.moveTo(ex,ey);
          ctx.lineTo(ex+Math.cos(a1)*headLen, ey+Math.sin(a1)*headLen);
          ctx.lineTo(ex+Math.cos(a2)*headLen, ey+Math.sin(a2)*headLen);
          ctx.closePath(); ctx.fill();
          /* magnitude label */
          ctx.font = "11px 'Segoe UI',sans-serif";
          ctx.fillStyle = "rgba(109,213,250,0.8)";
          ctx.textAlign = "center";
          ctx.fillText(mag.toFixed(1)+" px", ax, ay-14);
          ctx.restore();
        }
      }
    }

    /* -- GT comparison indicator -- */
    if(HAS_GT && tGt && tGt.checked && idx > 0 && idx <= GAZE.length){
      var g = GAZE[Math.min(idx, GAZE.length-1)];
      if(g.gt !== undefined && Math.abs(g.t - currentT) < 0.1){
        var match = (g.ev === g.gt);
        ctx.save();
        ctx.fillStyle = match ? "rgba(34,197,94,0.7)" : "rgba(239,68,68,0.9)";
        ctx.font = "bold 11px 'Segoe UI',sans-serif";
        ctx.textAlign = "left";
        ctx.fillText(match ? "GT \u2713" : "GT \u2717", 12, 22);
        ctx.restore();
      }
    }

    /* Time display */
    var dur = vid.duration || 1;
    timeDisp.textContent = currentT.toFixed(2)+" / "+dur.toFixed(2)+" s";
    seekBar.value = Math.round((currentT/dur)*1000);

    drawPlayhead(currentT);
    requestAnimationFrame(drawOverlay);
  }

  /* -- Controls -- */
  btnPlay.addEventListener("click", function(){ vid.play(); });
  btnPause.addEventListener("click", function(){ vid.pause(); });

  var speedBtns = document.querySelectorAll("[data-speed]");
  for(var si=0;si<speedBtns.length;si++){
    (function(btn){
      btn.addEventListener("click", function(){
        vid.playbackRate = parseFloat(btn.dataset.speed);
        for(var j=0;j<speedBtns.length;j++) speedBtns[j].classList.remove("active");
        btn.classList.add("active");
      });
    })(speedBtns[si]);
  }

  seekBar.addEventListener("input", function(){
    vid.currentTime = (seekBar.value/1000) * (vid.duration||0);
  });

  /* -- Stats cards -- */
  function renderStats(){
    var html = "";
    html += card("Total Gaze Points", STATS.n_fixations);
    html += card("Fixation Centers", STATS.n_fixation_points);
    html += card("Saccade Points", STATS.n_saccade_points);
    html += card("Duration", STATS.duration_s+" s");
    if(STATS.f1_fixation !== undefined){
      html += card("F1 Fixation", STATS.f1_fixation, STATS.f1_fixation>=0.90);
      html += card("F1 Saccade", STATS.f1_saccade, STATS.f1_saccade>=0.50);
    }
    statsArea.innerHTML = html;
  }
  function card(label, value, good){
    var color = good===undefined ? "#6dd5fa" : (good ? "#4caf50" : "#ff9800");
    return '<div class="stat-box" style="border-left-color:'+color+'">'+
      '<div class="value" style="color:'+color+'">'+value+'</div>'+
      '<div class="label">'+label+'</div></div>';
  }
  renderStats();

  /* -- Start loop -- */
  requestAnimationFrame(drawOverlay);
})();
</script>
</body>
</html>"""
