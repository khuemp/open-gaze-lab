import plotly.graph_objs as go
import plotly.offline as pyo

from ._image_utils import encode_image_base64


def plot_gaze_points_and_fixations(self, output_dir, bg_image_path=None, aois=None, show_attach=True,
                                    attach_type='bbox', y_origin='top-left'):
    """Creates an interactive visualization of gaze data, fixations, and AOIs.

    Generates a Plotly-based interactive plot showing gaze samples, fixations, and
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

    # gaze samples scatter
    gaze_scatter = go.Scatter(
        x=gaze_data['x'],
        y=gaze_data['y'],
        mode='markers',
        marker=dict(color=colors, size=6, opacity=0.7),
        name='Gaze Samples'
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
        name='Fixation Events'
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
