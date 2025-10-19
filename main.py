import pandas as pd
import os
from event_detection import EventDetection, EyeTrackingVisualizer

if __name__ == '__main__':
    script_dir = os.path.abspath(os.path.dirname(__file__))
    input_folder = os.path.join(script_dir, 'gaze_data')
    event_folder = os.path.join(script_dir, 'event_results')
    plots_folder = os.path.join(script_dir, 'plots')

    if not os.path.isdir(input_folder):
        raise FileNotFoundError(f"Input folder not found: {input_folder}")

    os.makedirs(event_folder, exist_ok=True)
    os.makedirs(plots_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.csv'):
            csv_path = os.path.join(input_folder, filename)

            try:
                # This loop now also handles the timestamp conversion.
                for delim in [';', ',', '\t', ' ']:
                    gaze_file = pd.read_csv(csv_path, delimiter=delim)
                    if gaze_file.shape[1] > 1:  # Check if delimiter worked
                        
                        # If the delimiter was a comma, convert the timestamp
                        if delim == ',':
                            gaze_file['timestamp'] /= 1000.0
                        
                        break # Exit loop once the correct delimiter is found

                # Add resolution scaling here
                gaze_file['x'] *= 2560  
                gaze_file['y'] *= 1440  

                event_detection = EventDetection(gaze_file)

                event_output_path = os.path.join(event_folder, f"{os.path.splitext(filename)[0]}_events.csv")

                event_results = event_detection.process_event(
                    output_dir=event_output_path,
                    min_fixation_duration=100.0/1000.0,  # in seconds
                    fixation_merge_threshold=100.0,
                    detect_threshold=25.0, # in pixels
                    adapt=False,
                    optimize=False,
                    algorithm='idt'
                )

                plot_output_path = os.path.join(plots_folder, f"{os.path.splitext(filename)[0]}_plot.html")

                event_visualizer = EyeTrackingVisualizer(event_results)
                event_visualizer.plot_gaze_points_and_fixations(
                    output_dir=plot_output_path,
                    bg_image_path=None,
                    aois=None,
                    show_attach=True,
                    attach_type='bbox'
                )

            except Exception as e:
                print(f"⚠️ Error processing {filename}: {e}")