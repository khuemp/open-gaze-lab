
import plotly.graph_objs as go
import plotly.offline as pyo

def plot_gaze_points_and_fixations(self, output_dir, bg_image_path=None, aois=None, show_attach=True,
                                    attach_type='bbox'):
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

    Returns:
        None: Saves an interactive HTML plot to the specified output directory
    
    Notes:
        Requires self.event_data_df to contain:
            - x, y: Raw gaze coordinates
            - fixation_x, fixation_y: Fixation center coordinates
            - event_type: 'Fixation' or 'Saccade'
    """

    gaze_data = self.event_data_df.copy()

    # Define visual style for event types
    color_map = {'Fixation': 'rgba(0, 128, 0, 0.5)',    # Green for fixations
                 'Saccade': 'rgba(128, 0, 0, 0.5)'}     # Red for saccades
    colors = [color_map[event] for event in gaze_data['event_type']]

    # Create scatter plot of gaze points colored by event type (fixation/saccade)
    gaze_scatter = go.Scatter(
        x=gaze_data['x'],
        y=gaze_data['y'],
        mode='markers',
        marker=dict(color=colors),     # Colors defined by event type
        name='Gaze Points'
    )

    # Filter and prepare fixation center data
    fixations = gaze_data[['fixation_x', 'fixation_y']].drop_duplicates()
    fixations = fixations[fixations['fixation_x'].notna() & 
                         fixations['fixation_y'].notna()]    # Exclude invalid coordinates
    fixations.reset_index(drop=True, inplace=True)          # Reset index for sequential labeling

    # Create visualization of fixation centers with numbered labels
    fixation_scatter = go.Scatter(
        x=fixations['fixation_x'],
        y=fixations['fixation_y'],
        mode='markers+text',
        marker=dict(color='white', size=10, 
                   line=dict(color='black', width=2)),      # High-contrast markers
        text=[str(index) for index in fixations.index],     # Chronological numbering
        textposition='top center',                          # Label position
        name='Fixation Points'
    )

    # Draw scanpath connecting fixations in temporal order
    fixation_line = go.Scatter(
        x=fixations['fixation_x'],
        y=fixations['fixation_y'],
        mode='lines',
        line=dict(color='black', width=2),
        name='Fixation Path'
    )

    # Configure plot appearance and axes
    layout = go.Layout(
        title='Gaze and Fixation Visualization',
        xaxis=dict(title='X', range=[0, 2560]),        # Full HD width
        yaxis=dict(title='Y', range=[1440, 0],         # Inverted Y-axis for screen coordinates
                  autorange=False),
        showlegend=True
    )

    # Combine visualization layers
    data = [gaze_scatter, fixation_scatter, fixation_line]
    fig = go.Figure(data=data, layout=layout)

    # Add optional background stimulus image
    if bg_image_path is not None:
        # Load background image data
        image = open(bg_image_path, 'rb').read()

        # Add background image as base layer with controlled opacity
        fig.add_layout_image(
            dict(
                source=image,
                xref="x", yref="y",           # Align with plot coordinates
                x=0, y=0,                     # Position at origin
                sizex=2560, sizey=1440,       # Match screen dimensions
                sizing="stretch",             # Scale to fit
                opacity=0.5,                  # Semi-transparent for overlay
                layer="below")                # Place behind other elements
        )

    if aois is not None:
        # Visualize Areas of Interest (AOIs) as semi-transparent rectangles
        for index, aoi in aois.iterrows():
            # Create AOI bounding box
            fig.add_shape(
                type="rect",                               # Rectangle shape for AOI boundary
                xref="x", yref="y",                       # Reference to plot coordinates
                x0=aoi['pos_x'],                          # Left edge
                y0=aoi['pos_y'],                          # Top edge
                x1=aoi['pos_x'] + aoi['width'],           # Right edge
                y1=aoi['pos_y'] + aoi['height'],          # Bottom edge
                line=dict(
                    color="RoyalBlue",                    # Blue border
                    width=3,                              # Prominent border width
                ),
                fillcolor="LightSkyBlue",                 # Light fill color
                opacity=0.5,                              # Semi-transparent fill
                layer="below"                             # Draw under gaze data
            )

            # Add centered AOI identification labels
            fig.add_annotation(
                x=aoi['pos_x'] + aoi['width'] / 2,        # Horizontal center
                y=aoi['pos_y'] + aoi['height'] / 2,       # Vertical center
                text=f"{aoi['aoi']}",                     # AOI identifier
                showarrow=False,                          # No pointer arrow
                font=dict(
                    color="RoyalBlue",                    # Match border color
                    size=12                               # Readable text size
                )
            )

    # Visualize fixation-to-AOI attachments when enabled
    if show_attach and 'aoi' in gaze_data.columns and aois is not None:
        # Extract fixations that have been assigned to AOIs
        fixations = gaze_data[gaze_data['aoi_type'].notna()][
            ['fixation_x', 'fixation_y', 'aoi_type', 'aoi', 'aoi_id']].drop_duplicates()
        fixations.reset_index(drop=True, inplace=True)

        # Draw connection lines between fixations and their assigned AOIs
        for idx, fixation in fixations.iterrows():
            # Only process word-type AOIs for attachment visualization
            if fixation['aoi_type'] == 'word':
                # Get the corresponding AOI and calculate its center point
                aoi = aois.iloc[int(fixation['aoi_id'])]
                aoi_center_x = aoi['pos_x'] + aoi['width'] / 2    # Horizontal center
                aoi_center_y = aoi['pos_y'] + aoi['height'] / 2   # Vertical center
            else:
                continue

            # Create visual connection between fixation and AOI center
            line = go.Scatter(
                x=[fixation['fixation_x'], aoi_center_x],     # Line endpoints X
                y=[fixation['fixation_y'], aoi_center_y],     # Line endpoints Y
                mode='lines',
                line=dict(color='red', width=2),              # Red connection line
                name=f'Fixation {idx} to AOI Centroid'        # Identify connection
            )
            fig.add_trace(line)

    # Save the figure to an HTML file
    pyo.plot(fig, filename=output_dir, auto_open=False)
