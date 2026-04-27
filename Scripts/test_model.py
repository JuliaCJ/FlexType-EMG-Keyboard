import os
import glob
import re
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Parameter Configuration
fs = 500
MODEL_WINDOW_SIZE = 125  # The model expects 125 samples (250ms)
MODEL_STEP_SIZE = 62
RMS_WINDOW_MS = 100  # User requested 100ms for RMS plots
main_folder = 'CPE4850 - Gesture Data/Test Data'

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
    "Gesture 3": "Middle Pinch", # a-f
    "Gesture 4": "Ring Pinch", # g-m
    "Gesture 5": "Pinky Pinch", # m-t
    "Gesture 6": "L-Sign", # swipe
    "Gesture 7": "Thumb-Out", # delete
    "Gesture 8": "Knock", # space/ enter
    #"Gesture 9": "Pinky Up",
    "Gesture 10": "Three Fingers", # u-z
    #"Gesture 11": "Wiggle Fingers",
    "Gesture 12": "Surfs Up" # start/ stop
}

NUM_CLASSES = len(gesture_mapping)
ordered_class_names = [gesture_mapping[g] for g in gesture_mapping.keys()]

# Helper Functions
class GestureVoter:
    def __init__(self, num_classes, decay=0.2, thresholds=None):
        self.ema_probs = np.zeros(num_classes)
        self.decay = decay
        self.current_stable = 0
        self.initialized = False
        self.default_threshold = 0.45
        self.thresholds = thresholds if thresholds else {}

    def predict_voted_gesture(self, new_probs, temperature=0.8):
        sharpened_probs = np.power(new_probs, 1 / temperature)
        sharpened_probs /= np.sum(sharpened_probs)
        if not self.initialized:
            self.ema_probs = sharpened_probs
            self.initialized = True
        else:
            self.ema_probs = (1 - self.decay) * self.ema_probs + self.decay * sharpened_probs

        best_gesture = np.argmax(self.ema_probs)
        best_confidence = self.ema_probs[best_gesture]
        thresh = self.thresholds.get(best_gesture, self.default_threshold)

        if best_gesture != self.current_stable:
            if best_confidence >= thresh:
                self.current_stable = best_gesture
        else:
            if self.current_stable != 0:
                if self.ema_probs[0] > 0.6:
                    self.current_stable = 0
        return self.current_stable


def segment_raw_windows(signal, window_size, step_size):
    windows = []
    for start in range(0, signal.shape[0] - window_size + 1, step_size):
        windows.append(signal[start:start + window_size, :])
    return np.array(windows)


def calculate_rms(window):
    return np.sqrt(np.mean(np.square(window), axis=0))


def segment_and_rms(signal, window_size, step_size):
    rms_values = []
    for start in range(0, signal.shape[0] - window_size + 1, step_size):
        window = signal[start:start + window_size, :]
        rms_values.append(calculate_rms(window))
    return np.array(rms_values)


def plot_rms():
    plot_window_size = int((RMS_WINDOW_MS / 1000) * fs)
    plot_step_size = plot_window_size // 2

    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharey=True)
    axes = axes.flatten()

    for i, (folder_name, label) in enumerate(gesture_mapping.items()):
        folder_path = os.path.join(main_folder, folder_name)
        files = glob.glob(os.path.join(folder_path, '*.csv'))
        if not files: continue

        df = pd.read_csv(files[0], sep=r'\s+', header=0)
        emg_data = df.filter(like='FilteredChannel').values.astype(np.float32)
        if emg_data.shape[1] == 0: emg_data = df.iloc[:, :8].values.astype(np.float32)

        # Trimming for gestures
        if folder_name != "Gesture 0":
            emg_data = emg_data[2 * fs:-2 * fs] if len(emg_data) > 4 * fs else emg_data

        rms_data = segment_and_rms(emg_data, plot_window_size, plot_step_size)
        time_axis = np.arange(len(rms_data)) * (plot_step_size / fs)

        for channel in range(8):
            axes[i].plot(time_axis, rms_data[:, channel], alpha=0.7)
        axes[i].set_title(label)
        axes[i].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.suptitle(f"RMS Values ({RMS_WINDOW_MS}ms Windows) per Gesture", y=1.02)
    os.makedirs('Results', exist_ok=True)
    plt.savefig('Results/gesture_rms_plots.png')
    plt.show()

# Testing Loop
model = load_model('gesture_recognition_model.keras')
all_predictions, true_labels = [], []

for folder_name in [f for f in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, f))]:
    if folder_name not in gesture_mapping: continue

    gesture_idx = list(gesture_mapping.keys()).index(folder_name)
    folder_path = os.path.join(main_folder, folder_name)

    for file in glob.glob(os.path.join(folder_path, '*.csv')):
        voter = GestureVoter(NUM_CLASSES, decay=EMA_DECAY, thresholds=CUSTOM_THRESHOLDS)
        df = pd.read_csv(file, sep=r'\s+', header=0)

        emg_data = df.filter(like='FilteredChannel').values.astype(np.float32)
        if emg_data.shape[1] == 0: emg_data = df.iloc[:, :8].values.astype(np.float32)

        if folder_name != "Gesture 0":
            emg_data = emg_data[2 * fs:-2 * fs] if len(emg_data) > 4 * fs else emg_data

        windows = segment_raw_windows(emg_data, MODEL_WINDOW_SIZE, MODEL_STEP_SIZE)

        if len(windows) > 0:
            preds = model.predict(windows, verbose=0)
            for p in preds:
                stable_pred = voter.predict_voted_gesture(p, temperature=PROB_TEMP)
                all_predictions.append(stable_pred)
                true_labels.append(gesture_idx)

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

# RMS PLot
plot_rms()