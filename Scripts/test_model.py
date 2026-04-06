import os
import glob
import re
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report

# Configuration
fs = 500
window_size = 125
step_size = 62

# Gesture mapping
gesture_mapping = {
    "Gesture 0": "Rest",
    "Gesture 1": "Two-Finger Wave",
    "Gesture 3": "Middle Pinch",
    "Gesture 4": "Ring Pinch",
    "Gesture 5": "Pinky Pinch",
    "Gesture 6": "L-Sign",
    "Gesture 7": "Thumb-Out",
    "Gesture 8": "Knock",
    "Gesture 11": "Wiggle Fingers"
}

# Helper functions
def segment(signal, window_size=125, step_size=62):
    segments = []
    for start in range(0, signal.shape[0] - window_size + 1, step_size):
        window = signal[start:start + window_size, :]
        segments.append(window)
    return np.array(segments)

def extract_number(name):
    match = re.search(r'\d+', name)
    return int(match.group()) if match else 0

# Names for classification
ordered_class_names = [gesture_mapping[g] for g in gesture_mapping.keys()]

# Setup
model = load_model('gesture_recognition_model.keras')
main_folder = 'CPE4850 - Gesture Data/Test Subject'
gesture_folders = [f for f in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, f))]

all_predictions = []
true_labels = []

# Prediction Loop
for folder_name in gesture_folders:
    # Skip excluded gestures
    if folder_name not in gesture_mapping:
        continue

    # Get the numeric index (0-8) based on the mapping keys
    current_gesture_idx = list(gesture_mapping.keys()).index(folder_name)
    real_name = gesture_mapping[folder_name]

    print(f"Testing {folder_name} ({real_name})...")

    folder_path = os.path.join(main_folder, folder_name)
    for file in glob.glob(os.path.join(folder_path, '*.csv')):
        # Read the CSV using the same separator as training
        df = pd.read_csv(file, sep=r'\s+', header=0)

        # Select the specific 8 channels the model expects
        filtered_cols = [
            'FilteredChannel1', 'FilteredChannel2', 'FilteredChannel3', 'FilteredChannel4',
            'FilteredChannel5', 'FilteredChannel6', 'FilteredChannel7', 'FilteredChannel8'
        ]

        try:
            emg_data = df[filtered_cols].values.astype(np.float32)
        except KeyError:
            # Fallback if headers are missing: grab first 8 columns
            emg_data = df.iloc[:, :8].values.astype(np.float32)

        # Trim 2 seconds on start/end data
        cut_sample_rest = int(2 * fs)
        if emg_data.shape[0] > 4 * fs and folder_name != "Gesture 0":
            emg_data = emg_data[cut_sample_rest:-cut_sample_rest]

        # Slice the data into windows
        windows = segment(emg_data, window_size=window_size, step_size=step_size)
        if windows.shape[0] > 0:
            preds = model.predict(windows, verbose=0)
            for p in preds:
                all_predictions.append(np.argmax(p))
                true_labels.append(current_gesture_idx)

# Accuracy report
print("\n--- Final Results ---")
print(classification_report(
    true_labels,
    all_predictions,
    target_names=ordered_class_names
))