import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configuration
fs = 500
RMS_WINDOW_MS = 100  # ms window for RMS plotting
main_folder = 'CPE4850 - Gesture Data/Test Data'

gesture_mapping = {
    "Gesture 0": "Rest",
    "Gesture 3": "Middle Pinch",
    "Gesture 4": "Ring Pinch",
    "Gesture 5": "Pinky Pinch",
    "Gesture 6": "L-Sign",
    "Gesture 7": "Thumb-Out",
    "Gesture 8": "Knock",
    "Gesture 10": "Three Fingers",
    "Gesture 12": "Surfs Up"
}

# Processing Functions
def calculate_rms(window):
    return np.sqrt(np.mean(np.square(window), axis=0))

def segment_and_rms(signal, window_size, step_size):
    rms_values = []
    for start in range(0, signal.shape[0] - window_size + 1, step_size):
        window = signal[start:start + window_size, :]
        rms_values.append(calculate_rms(window))
    return np.array(rms_values)


def get_centered_data(file_path, folder_name):
    df = pd.read_csv(file_path, sep=r'\s+', header=0)
    emg_data = df.filter(like='FilteredChannel').values.astype(np.float32)
    if emg_data.shape[1] == 0:
        emg_data = df.iloc[:, :8].values.astype(np.float32)

    target_len = 3 * fs
    if emg_data.shape[0] >= target_len and folder_name != "Gesture 0":
        start_idx = (emg_data.shape[0] - target_len) // 2
        emg_data = emg_data[start_idx:start_idx + target_len]
    return emg_data

# Plotting Functions
def plot_time_domain():
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    axes = axes.flatten()

    for i, (folder_name, label) in enumerate(gesture_mapping.items()):
        folder_path = os.path.join(main_folder, folder_name)
        files = glob.glob(os.path.join(folder_path, '*.csv'))
        if not files: continue

        emg_data = get_centered_data(files[0], folder_name)
        time_axis = np.linspace(0, len(emg_data) / fs, num=len(emg_data))

        for channel in range(8):
            axes[i].plot(time_axis, emg_data[:, channel], alpha=0.8, linewidth=0.7)

        axes[i].set_title(f"{label}", fontsize=13, pad=10)
        axes[i].grid(True, alpha=0.2)
        axes[i].set_xlabel("Time (s)")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    plt.suptitle("EMG Time Domain Signals", fontsize=20, weight='bold')
    os.makedirs('Results', exist_ok=True)
    plt.savefig('Results/time_domain_comparison.png')
    plt.show()


def plot_rms_domain():
    plot_window_size = int((RMS_WINDOW_MS / 1000) * fs)  # 50 samples
    plot_step_size = plot_window_size // 2

    fig, axes = plt.subplots(3, 3, figsize=(18, 14), sharey=True)
    axes = axes.flatten()

    for i, (folder_name, label) in enumerate(gesture_mapping.items()):
        folder_path = os.path.join(main_folder, folder_name)
        files = glob.glob(os.path.join(folder_path, '*.csv'))
        if not files: continue

        emg_data = get_centered_data(files[0], folder_name)
        rms_data = segment_and_rms(emg_data, plot_window_size, plot_step_size)
        time_axis = np.arange(len(rms_data)) * (plot_step_size / fs)

        for channel in range(8):
            axes[i].plot(time_axis, rms_data[:, channel], alpha=0.7)

        axes[i].set_title(f"{label} ({RMS_WINDOW_MS}ms RMS)", fontsize=13, pad=10)
        axes[i].set_xlabel("Time (s)")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(hspace=0.6, wspace=0.3)

if __name__ == "__main__":
    plot_time_domain()
    plot_rms_domain()