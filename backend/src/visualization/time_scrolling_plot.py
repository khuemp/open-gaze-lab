import plotly.graph_objs as go
import plotly.offline as pyo

from ._image_utils import encode_image_base64


def plot_gaze_with_time_scrolling(self, output_dir, bg_image_path=None, aois=None,
                                   time_window_ms=5000, step_ms=100, y_origin='top-left'):
    """Creates an interactive time-scrollable visualization of gaze data and fixations.

    Generates a Plotly-based animated plot that allows users to scroll through time
    and see how gaze samples and fixations appear progressively. Includes a time slider
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
            name='Gaze Samples',
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
            name='Fixation Events',
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
            image_source = encode_image_base64(bg_image_path)
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
