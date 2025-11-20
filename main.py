import pandas as pd
import os
from event_detection import EventDetection, EyeTrackingVisualizer

if __name__ == '__main__':

    script_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(script_dir, 'data')
    folders = {
        'gaze': os.path.join(data_dir, 'gaze_data'),
        'event': os.path.join(data_dir, 'event_data'),
        'scanpath': os.path.join(data_dir, 'scanpath_data')
    }

    if not os.path.isdir(folders['gaze']):
        raise FileNotFoundError(f"Input folder not found: {folders['gaze']}")

    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)

    for filename in os.listdir(folders['gaze']):
        if filename.lower().endswith('.csv'):
            csv_path = os.path.join(folders['gaze'], filename)

            try:
                # This loop now also handles the timestamp conversion.
                for delim in [';', ',', '\t', ' ']:
                    gaze_data = pd.read_csv(csv_path, delimiter=delim)
                    if gaze_data.shape[1] > 1:  # Check if delimiter worked
                        break # Exit loop once the correct delimiter is found

                event_detection = EventDetection(gaze_data, resolution=(2560, 1440))

                event_output_path = os.path.join(folders['event'], f"{os.path.splitext(filename)[0]}.csv")

                event_results = event_detection.process_event(
                    output_dir=event_output_path,
                    min_fixation_duration=100.0,  # in milliseconds
                    fixation_merge_threshold=None,  # in pixels
                    detect_threshold=125.0, # in pixels
                    adapt=False,
                    algorithm='idt',
                    sampling_rate=30 # Hz
                )

                plot_output_path = os.path.join(folders['scanpath'], f"{os.path.splitext(filename)[0]}.html")

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