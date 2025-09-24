
import pandas as pd

from event_detection import EventDetection

if __name__ == '__main__':
    # TODO: Enable reading different input files
    # Process a single file: A01_1966_False_False.csv located at project root.
    import os
    # Build absolute path to CSV located in the same directory as this script
    script_dir = os.path.abspath(os.path.dirname(__file__))
    csv_path = os.path.join(script_dir, 'gaze.csv')

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Input file not found: {csv_path}")

    print(f"Reading gaze data from: {csv_path}")
    # Adjust delimiter if necessary (currently assumes semicolon)
    gaze_file = pd.read_csv(csv_path, delimiter=';')
    event_detection = EventDetection(gaze_file)

    # Output path (same directory, different name)
    output_path = os.path.join(script_dir, 'fixations.csv')

    event_detection.process_event_with_merge(output_dir=output_path,
                                             plot=False,
                                             min_fixation_duration=50.0,
                                             merge_distance=None,
                                             threshold=25.0,
                                             adapt=False,
                                             optimize_threshold=False,
                                             algorithm='idt')

    print(f"Processing complete! Output written to {output_path}")