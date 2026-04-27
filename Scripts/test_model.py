import os
import glob
import re
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from collections import deque, Counter

# Voter Class
class GestureVoter:
    def __init__(self, num_classes, decay=0.2, thresholds=None):
        self.ema_probs = np.zeros(num_classes)
        self.decay = decay
        self.current_stable = 0
        self.initialized = False

        # Default threshold if not specified
        self.default_threshold = 0.45

        # Map specific thresholds to gestures that are confused often
        self.thresholds = thresholds if thresholds else {}

    def predict_voted_gesture(self, new_probs, temperature=0.8):
        # Softmax temperature sharpening
        sharpened_probs = np.power(new_probs, 1 / temperature)
        sharpened_probs /= np.sum(sharpened_probs)

        if not self.initialized:
            self.ema_probs = sharpened_probs
            self.initialized = True
        else:
            # EMA Update
            self.ema_probs = (1 - self.decay) * self.ema_probs + self.decay * sharpened_probs

        best_gesture = np.argmax(self.ema_probs)
        best_confidence = self.ema_probs[best_gesture]

        # Get threshold for gestures
        thresh = self.thresholds.get(best_gesture, self.default_threshold)

        # State transitions
        if best_gesture != self.current_stable:
            # Only switch if threshold met
            if best_confidence >= thresh:
                self.current_stable = best_gesture
        else:
            # Only switch to rest from gesture prediction if probability is high.
            if self.current_stable != 0:
                rest_prob = self.ema_probs[0]
                # Rest prediction must be > 60%
                if rest_prob > 0.6:
                    self.current_stable = 0

        return self.current_stable

# Configuration
fs = 500
window_size = 125
step_size = 62
VOTE_BUFFER_SIZE = 5

# Tuning Dictionary
CUSTOM_THRESHOLDS = {
    1:  0.55, # Two-Finger Wave
    3:  0.40, # Middle Pinch
    4:  0.40, # Ring Pinch
    9:  0.65, # Pinky Up
    11: 0.35, # Wiggle Fingers
}

# Voter Tuning Parameters
EMA_DECAY = 0.3          # How much weight new predictions carry vs history
PROB_TEMP = 0.8
ENTER_THRESHOLD = 0.5   # Confidence needed to switch to a new gesture
EXIT_THRESHOLD = 0.2    # Confidence drop needed to revert to Rest

# Gesture mapping
gesture_mapping = {
    "Gesture 0": "Rest",
    #"Gesture 1": "Two-Finger Wave",
    #"Gesture 2": "Index Pinch",
    "Gesture 3": "Middle Pinch",
    "Gesture 4": "Ring Pinch",
    "Gesture 5": "Pinky Pinch",
    "Gesture 6": "L-Sign",
    "Gesture 7": "Thumb-Out",
    "Gesture 8": "Knock",
    #"Gesture 9": "Pinky Up",
    "Gesture 10": "Three Fingers",
    #"Gesture 11": "Wiggle Fingers",
    "Gesture 12": "Surfs Up"
}

NUM_CLASSES = len(gesture_mapping)

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

ordered_class_names = [gesture_mapping[g] for g in gesture_mapping.keys()]

# Setup
model = load_model('gesture_recognition_model.keras')
main_folder = 'CPE4850 - Gesture Data/Test Data'
gesture_folders = [f for f in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, f))]

all_predictions = []
true_labels = []

# Prediction Loop
for folder_name in gesture_folders:
    if folder_name not in gesture_mapping:
        continue

    current_gesture_idx = list(gesture_mapping.keys()).index(folder_name)
    real_name = gesture_mapping[folder_name]

    print(f"Testing {folder_name} ({real_name})...")

    folder_path = os.path.join(main_folder, folder_name)
    for file in glob.glob(os.path.join(folder_path, '*.csv')):

        # Instantiate a fresh voter for every new file/sequence
        voter = GestureVoter(
            num_classes=NUM_CLASSES,
            decay=EMA_DECAY,
            thresholds=CUSTOM_THRESHOLDS
        )

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
        cut_sample_rest = int(2*fs)
        if emg_data.shape[0] > 4 * fs and folder_name != "Gesture 0":
            emg_data = emg_data[cut_sample_rest:-cut_sample_rest]

        # Slice the data into windows
        windows = segment(emg_data, window_size=window_size, step_size=step_size)

        if windows.shape[0] > 0:
            preds = model.predict(windows, verbose=0)

            for p in preds:
                # Pass the temperature parameter to sharpen predictions
                stable_pred = voter.predict_voted_gesture(p, temperature=PROB_TEMP)
                all_predictions.append(stable_pred)
                true_labels.append(current_gesture_idx)

# Accuracy report
print("\n--- Final Results ---")
final_accuracy = accuracy_score(true_labels, all_predictions)
print(f"Overall Accuracy: {final_accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    true_labels,
    all_predictions,
    target_names=ordered_class_names
))

# Generate Confusion Matrix
cm = confusion_matrix(true_labels, all_predictions)

# Plotting the Confusion Matrix
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=ordered_class_names,
            yticklabels=ordered_class_names)
plt.title(f'Confusion Matrix (Accuracy: {final_accuracy * 100:.2f}%)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()

# Ensure the Results directory exists before saving
os.makedirs('Results', exist_ok=True)
plt.savefig('Results/test_conf_matrix.png')
plt.show()

class_accuracies = cm.diagonal() / cm.sum(axis=1)

print("\n--- Individual Gesture Accuracy ---")
for name, acc in zip(ordered_class_names, class_accuracies):
    display_acc = acc * 100 if not np.isnan(acc) else 0.0
    print(f"{name:15} : {display_acc:.2f}%")

# Identifying Most Confusing Gestures
print("\n--- Error Analysis: Most Confusing Gestures ---")
print(f"{'Gesture':<20} | {'Most Confused With':<20} | {'Error Count'}")
print("-" * 55)

for i, name in enumerate(ordered_class_names):
    row = cm[i].copy()
    row[i] = 0 # Set correct prediction count to zero to find highest error

    max_error_idx = np.argmax(row)
    max_error_count = row[max_error_idx]

    if max_error_count > 0:
        confused_with = ordered_class_names[max_error_idx]
        print(f"{name:<20} | {confused_with:<20} | {max_error_count}")
    else:
        print(f"{name:<20} | {'None (100% Correct)':<20} | 0")