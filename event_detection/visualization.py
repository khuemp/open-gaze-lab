
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

    # Create a color map based on classification
    color_map = {'Fixation': 'rgba(0, 128, 0, 0.5)', 'Saccade': 'rgba(128, 0, 0, 0.5)'}
    colors = [color_map[event] for event in gaze_data['event_type']]

    # Create scatter plots for gaze points
    gaze_scatter = go.Scatter(
        x=gaze_data['x'],
        y=gaze_data['y'],
        mode='markers',
        marker=dict(color=colors),
        name='Gaze Points'
    )

    # Get all unique fixation points
    fixations = gaze_data[['fixation_x', 'fixation_y']].drop_duplicates()
    # remove NaN values
    fixations = fixations[fixations['fixation_x'].notna() & fixations['fixation_y'].notna()]
    # reset index
    fixations.reset_index(drop=True, inplace=True)

    # Create a scatter plot for fixation points
    fixation_scatter = go.Scatter(
        x=fixations['fixation_x'],
        y=fixations['fixation_y'],
        mode='markers+text',
        marker=dict(color='white', size=10, line=dict(color='black', width=2)),
        text=[str(index) for index in fixations.index],
        textposition='top center',
        name='Fixation Points'
    )

    # Connect fixation points with a line
    fixation_line = go.Scatter(
        x=fixations['fixation_x'],
        y=fixations['fixation_y'],
        mode='lines',
        line=dict(color='black', width=2),
        name='Fixation Path'
    )

    # Set the layout of the plot
    layout = go.Layout(
        title='Gaze and Fixation Visualization',
        xaxis=dict(title='X', range=[0, 2560]),
        yaxis=dict(title='Y', range=[1440, 0], autorange=False),
        showlegend=True
    )

    # Combine all the elements to create the figure
    data = [gaze_scatter, fixation_scatter, fixation_line]
    fig = go.Figure(data=data, layout=layout)

    # Add background image if provided
    if bg_image_path is not None:
        # Read the image file
        image = open(bg_image_path, 'rb').read()

        # Add the image to the figure
        fig.add_layout_image(
            dict(
                source=image,
                xref="x",
                yref="y",
                x=0,
                y=0,
                sizex=2560,
                sizey=1440,
                sizing="stretch",
                opacity=0.5,
                layer="below")
        )

    if aois is not None:
        # Add AOIs bounding boxes
        for index, aoi in aois.iterrows():
            fig.add_shape(
                # Rectangle reference to the axes
                type="rect",
                xref="x",
                yref="y",
                x0=aoi['pos_x'],
                y0=aoi['pos_y'],
                x1=aoi['pos_x'] + aoi['width'],
                y1=aoi['pos_y'] + aoi['height'],
                line=dict(
                    color="RoyalBlue",
                    width=3,
                ),
                fillcolor="LightSkyBlue",
                opacity=0.5,
                layer="below"
            )

            # Add AOI labels
            fig.add_annotation(
                x=aoi['pos_x'] + aoi['width'] / 2,
                y=aoi['pos_y'] + aoi['height'] / 2,
                text=f"{aoi['aoi']}",
                showarrow=False,
                font=dict(
                    color="RoyalBlue",
                    size=12
                )
            )

    # Visualize the attach algorithm results if requested
    if show_attach and 'aoi' in gaze_data.columns and aois is not None:
        # Get fixations with AOI assignments
        fixations = gaze_data[gaze_data['aoi_type'].notna()][
            ['fixation_x', 'fixation_y', 'aoi_type', 'aoi', 'aoi_id']].drop_duplicates()
        fixations.reset_index(drop=True, inplace=True)

        # We will draw a line from each fixation to the centroid or bbox of the assigned AOI
        for idx, fixation in fixations.iterrows():
            # calculate the fixation to AOI centroid distance
            if fixation['aoi_type'] == 'word':
                aoi = aois.iloc[int(fixation['aoi_id'])]
                aoi_center_x = aoi['pos_x'] + aoi['width'] / 2
                aoi_center_y = aoi['pos_y'] + aoi['height'] / 2
            else:
                continue

            # Create a line from fixation to AOI centroid
            line = go.Scatter(
                x=[fixation['fixation_x'], aoi_center_x],
                y=[fixation['fixation_y'], aoi_center_y],
                mode='lines',
                line=dict(color='red', width=2),
                name=f'Fixation {idx} to AOI Centroid'
            )
            fig.add_trace(line)

    # Save the figure to an HTML file
    pyo.plot(fig, filename=output_dir, auto_open=False)
