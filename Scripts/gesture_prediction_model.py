import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import scipy
import pandas as pd
from scipy import integrate
from scipy import signal
from scipy.signal import butter, lfilter
import numpy as np
import csv
import os
import sys
import enum
from enum import Enum
import math
import matplotlib
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import (
    Dense,
    Conv1D,
    Conv2D,
    MaxPooling1D,
    AveragePooling1D,
    MaxPooling2D,
    Flatten,
    Dropout,
    Normalization,
    LayerNormalization,
    BatchNormalization,
    GlobalAveragePooling1D,
)
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from keras.models import load_model
from collections import deque, Counter

import seaborn as sns
import re

import torch

#Versions of all the libraries being used
print("TensorFlow:", tf.__version__)
print("SciPy:", scipy.__version__)
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("Matplotlib", matplotlib.__version__)
print("Scikit-learn", sklearn.__version__)
print("Seaborn", sns.__version__)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU detected: {gpus}")
    try:
        # Prevent TensorFlow from consuming all VRAM at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
else:
    print("No GPU detected. Check your CUDA/cuDNN installation.")

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
                # Rest predi
                if rest_prob > 0.6:
                    self.current_stable = 0

        return self.current_stable

def augment_emg_data(X, Y, noise_std=0.01, scale_range=(0.8, 1.2)):
    X_aug = []
    Y_aug = []

    for i in range(len(X)):
        sample = X[i]
        label = Y[i]

        # Original
        X_aug.append(sample)
        Y_aug.append(label)

        # Add Gaussian Noise
        noise = np.random.normal(0, noise_std, sample.shape)
        X_aug.append(sample + noise)
        Y_aug.append(label)

        # Random Scaling (Signal Strength)
        factor = np.random.uniform(scale_range[0], scale_range[1])
        X_aug.append(sample * factor)
        Y_aug.append(label)

    return np.array(X_aug), np.array(Y_aug)

def build_model():
    model_cnn = Sequential()
    model_cnn.add(Conv1D(64, kernel_size=15, activation='relu', padding='same', input_shape=(
    window_size, len(filtered_cols))))  # (Sampling rate = 500Hz * Window length = 250ms, 8 channels)
    model_cnn.add(MaxPooling1D(2))

    model_cnn.add(Conv1D(64, kernel_size=9, activation='relu', padding='same'))
    model_cnn.add(BatchNormalization())
    model_cnn.add(MaxPooling1D(2))
    model_cnn.add(Dropout(0.4))

    model_cnn.add(Conv1D(64, kernel_size=5, activation='relu', padding='same'))
    model_cnn.add(BatchNormalization())
    model_cnn.add(MaxPooling1D(2))
    model_cnn.add(Dropout(0.4))

    model_cnn.add(GlobalAveragePooling1D())
    model_cnn.add(Dense(64, activation='relu'))
    model_cnn.add(Dense(num_gestures, activation='softmax'))

    model_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model_cnn.summary()
    return model_cnn


# New Data Setup
data_path = "CPE4850 - Gesture Data/New Data"
fs = 500
window_size = 125
step_size = 62
num_channels = 8
num_epochs = 250

# Tuning Parameters
EMA_DECAY = 0.3
PROB_TEMP = 0.8
CUSTOM_THRESHOLDS = {
    1: 0.55, 3: 0.40, 4: 0.40, 9: 0.65, 11: 0.35
}

X_all = []
Y_all = []
gesture_names = [
    "Gesture 0",  # Rest
    "Gesture 3",  # Middle Pinch
    "Gesture 4",  # Ring Pinch
    "Gesture 5",  # Pinky Pinch
    "Gesture 6",  # L-Sign
    "Gesture 7",  # Thumb-Out
    "Gesture 8",  # Knock
    "Gesture 10", # Three Fingers
    "Gesture 12"  # Surfs Up
]

class_labels = [
    "Rest", "Middle Pinch", "Ring Pinch", "Pinky Pinch",
    "L-Sign", "Thumb-Out", "Knock", "Three Fingers", "Surfs Up"
]

num_gestures = len(gesture_names)

def segment(signal, window_size=125, step_size=62):
    segments = []
    for start in range(0, signal.shape[0] - window_size + 1, step_size):
        window = signal[start:start + window_size, :]
        segments.append(window)
    return np.array(segments)


for subject_folder in sorted(os.listdir(data_path)):
    subject_path = os.path.join(data_path, subject_folder)
    if not os.path.isdir(subject_path):
        continue


    def extract_number(name):
        return int(re.search(r'\d+', name).group())


    for gesture_folder in sorted(os.listdir(subject_path), key=extract_number):

        if gesture_folder not in gesture_names:
            continue

        gesture_path = os.path.join(subject_path, gesture_folder)
        if not os.path.isdir(gesture_path):
            continue

        gesture_idx = gesture_names.index(gesture_folder)

        print(f"Processing {subject_folder} - Gesture Index {gesture_idx} === {gesture_folder}...")

        for file_name in os.listdir(gesture_path):
            if not file_name.endswith(".csv"):
                continue

            file_path = os.path.join(gesture_path, file_name)
            df = pd.read_csv(file_path, sep=r'\s+', header=0)

            filtered_cols = [
                'FilteredChannel1', 'FilteredChannel2', 'FilteredChannel3', 'FilteredChannel4',
                'FilteredChannel5', 'FilteredChannel6', 'FilteredChannel7', 'FilteredChannel8'
            ]

            emg_data = df[filtered_cols].values.astype(np.float32)

            cut_sample_rest = int(2*fs)  # 2 seconds of rest data

            if emg_data.shape[0] > 4 * fs and gesture_folder != "Gesture 0":  # makes sure that file is longer than 4 seconds and doesnt cut rest gesture
                emg_data = emg_data[cut_sample_rest:-cut_sample_rest]

            windows = segment(emg_data, window_size=window_size, step_size=step_size)

            if gesture_folder == "Gesture 0":
                # Only take a fraction of rest data to prevent bias
                windows = windows[:len(windows) // 4]

            labels = np.full((windows.shape[0],), gesture_idx)

            X_all.append(windows)
            Y_all.append(labels)

X_data = np.vstack(X_all)

Y_data = np.concatenate(Y_all)

Y_data = to_categorical(Y_data, num_classes=len(gesture_names))

X_train, X_val, Y_train, Y_val = train_test_split(X_data, Y_data, test_size=0.2, random_state=100, shuffle=True)

print(f"Original Training Set Size: {len(X_train)}")

# Augment Training Data
print("Augmenting Training Set...")
X_train, Y_train = augment_emg_data(X_train, Y_train)

print(f"Augmented Training Set Size: {len(X_train)}")


#Creating the cnn model
model_cnn = build_model()

history = model_cnn.fit(X_train, Y_train, batch_size=32, epochs=num_epochs, validation_data=(X_val, Y_val))

val_loss, val_accuracy = model_cnn.evaluate(X_val, Y_val, verbose=1)
print(f"Validation Accuracy: {val_accuracy*100:.2f}%")

train_loss, train_accuracy = model_cnn.evaluate(X_train, Y_train, verbose=1)
print(f"Training Accuracy: {train_accuracy*100:.2f}%")

model_cnn.save('gesture_recognition_model_trimmed.keras')
print("Model has been saved...")

#Plots of Accuracy and Loss
plt.figure()
plt.plot(history.history['loss'], label = 'Training Loss', color = 'blue')
plt.plot(history.history['val_loss'], label = 'Val Loss', color = 'Orange')
plt.title('CNN Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig("Results/model_loss.png")
plt.show()

plt.figure()
plt.plot(history.history['accuracy'], label = 'Training Accuracy', color = 'blue')
plt.plot(history.history['val_accuracy'], label = 'Val Accuracy', color = 'Orange')
plt.title('CNN Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("Results/model_acc.png")
plt.show()

# Gesture Prediction with Voting
voter = GestureVoter(buffer_size=10)
y_pred_voted = []

print("Predicting Validation Data with EMA Voting...")

raw_probs = model_cnn.predict(X_val)
y_true = np.argmax(Y_val, axis=1)
y_pred_voted = []

# Initialize voter
voter = GestureVoter(
    num_classes=num_gestures,
    decay=EMA_DECAY,
    thresholds=CUSTOM_THRESHOLDS
)

for p in raw_probs:
    # Use the same logic as the test script
    stable_pred = voter.predict_voted_gesture(p, temperature=PROB_TEMP)
    y_pred_voted.append(stable_pred)

y_pred_voted = np.array(y_pred_voted)

# Generate confusion matrix with voting applied
cm_voted = confusion_matrix(y_true, y_pred_voted)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_voted, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Gestures')
plt.ylabel('True Gestures')
plt.title('Gesture Recognition Confusion Matrix')
plt.savefig("Results/conf_matrix_training  vcv.png")
plt.show()

print(classification_report(y_true, y_pred_voted, target_names=class_labels))