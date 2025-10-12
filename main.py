import pandas as pd
import os
from event_detection import EventDetection

if __name__ == '__main__':
    script_dir = os.path.abspath(os.path.dirname(__file__))
    input_folder = os.path.join(script_dir, 'gaze_data')
    output_folder = os.path.join(script_dir, 'event_results')

    if not os.path.isdir(input_folder):
        raise FileNotFoundError(f"Input folder not found: {input_folder}")

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.csv'):
            csv_path = os.path.join(input_folder, filename)

            try:
                # This loop now also handles the timestamp conversion.
                for delim in [';', ',']:
                    gaze_file = pd.read_csv(csv_path, delimiter=delim)
                    if gaze_file.shape[1] > 1:  # Check if delimiter worked
                        
                        # If the delimiter was a comma, convert the timestamp
                        if delim == ',':
                            gaze_file['timestamp'] /= 1000.0
                        
                        break # Exit loop once the correct delimiter is found
                
                event_detection = EventDetection(gaze_file)

                output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_events.csv")

                event_detection.process_event(
                    output_dir=output_path,
                    plot=False,
                    min_fixation_duration=100.0/1000.0,  # in seconds
                    merge_distance=None,
                    threshold=25.0,
                    adapt=False,
                    optimize=False,
                    algorithm='idt'
                )

            except Exception as e:
                print(f"⚠️ Error processing {filename}: {e}")