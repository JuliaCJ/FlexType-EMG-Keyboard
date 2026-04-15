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

import seaborn as sns
import re
#Versions of all the libraries being used
print("TensorFlow:", tf.__version__)
print("SciPy:", scipy.__version__)
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("Matplotlib", matplotlib.__version__)
print("Scikit-learn", sklearn.__version__)
print("Seaborn", sns.__version__)

# New Data Setup
data_path = "CPE4850 - Gesture Data/NEW DATA"

fs = 500
window_size = 125
step_size = 62
num_channels = 8
num_gestures = 9

X_all = []
Y_all = []
gesture_names = ["Gesture 0", "Gesture 1", "Gesture 3", "Gesture 4",
                 "Gesture 5", "Gesture 6", "Gesture 7", "Gesture 8",
                 "Gesture 11"]


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

        if gesture_folder == "Gesture 10" or gesture_folder == "Gesture 12" or gesture_folder == "Gesture 2" or gesture_folder == "Gesture 9":
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
                'FilteredChannel5', 'FilteredChannel6', 'FilteredChannel7', 'FilteredChannel8', 'GyroX',
                'GyroY', 'GyroZ', 'AccX', 'AccY', 'AccZ', 'PPG1', 'PPG2'
            ]

            emg_data = df[filtered_cols].values.astype(np.float32)

            cut_sample_rest = int(1.5 * fs)  # 2 seconds of rest data

            if emg_data.shape[
                0] > 4 * fs and gesture_folder != "Gesture 0":  # makes sure that file is longer than 4 seconds and doesnt cut rest gesture
                emg_data = emg_data[cut_sample_rest:-cut_sample_rest]

            windows = segment(emg_data, window_size=window_size, step_size=step_size)

            labels = np.full((windows.shape[0],), gesture_idx)

            X_all.append(windows)
            Y_all.append(labels)

X_data = np.vstack(X_all)

Y_data = np.concatenate(Y_all)

Y_data = to_categorical(Y_data, num_classes=num_gestures)

X_train, X_val, Y_train, Y_val = train_test_split(X_data, Y_data, test_size=0.2, random_state=100, shuffle=True)

# #Augmenting Data Set
# def data_augment_noise(x, noise_std = 0.01):
#     noise = np.random.normal(0, noise_std, x.shape)
#     return x + noise
# X_train_augmented = []
# Y_train_augmented = []

# for i in range(len(X_train)):
#     x = X_train[i]
#     y = Y_train[i]

#     x_augmented = data_augment_noise(x)
#     X_train_augmented.append(x_augmented)
#     Y_train_augmented.append(y)

# X_train_augmented = np.array(X_train_augmented)
# Y_train_augmented = np.array(Y_train_augmented)

# X_train = np.concatenate([X_train, X_train_augmented], axis=0)
# Y_train = np.concatenate([Y_train, Y_train_augmented], axis=0)

#Creating the cnn model
model_cnn = Sequential()
model_cnn.add(Conv1D(64, kernel_size=15, activation = 'relu', padding='same', input_shape=(window_size, len(filtered_cols)))) #(Sampling rate = 500Hz * Window length = 250ms, 8 channels)
model_cnn.add(MaxPooling1D(2))

model_cnn.add(Conv1D(64, kernel_size=9, activation = 'relu', padding='same'))
model_cnn.add(BatchNormalization())
model_cnn.add(MaxPooling1D(2))
model_cnn.add(Dropout(0.4))

model_cnn.add(Conv1D(64, kernel_size=5, activation = 'relu', padding='same'))
model_cnn.add(BatchNormalization())
model_cnn.add(MaxPooling1D(2))
model_cnn.add(Dropout(0.4))

model_cnn.add(GlobalAveragePooling1D())
model_cnn.add(Dense(64, activation = 'relu'))
model_cnn.add(Dense(num_gestures, activation = 'softmax'))

model_cnn.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model_cnn.summary()

history = model_cnn.fit(X_train, Y_train, batch_size=32, epochs=100, validation_data=(X_val, Y_val))

val_loss, val_accuracy = model_cnn.evaluate(X_val, Y_val, verbose=1)
print(f"Validation Accuracy: {val_accuracy*100:.2f}%")

train_loss, train_accuracy = model_cnn.evaluate(X_train, Y_train, verbose=1)
print(f"Training Accuracy: {train_accuracy*100:.2f}%")

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

#Confusion Matrix for Gestures
y_pred_probs = model_cnn.predict(X_val)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(Y_val, axis=1)

cm = confusion_matrix(y_true, y_pred)
print(cm)
class_labels = ["Rest", "Two-Finger Wave", "Middle Pinch", "Ring Pinch",
                "Pinky Pinch", "L-Sign", "Thumb-Out", "Knock",
                "Wiggle Fingers"]

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Gestures')
plt.ylabel('True Gestures')
plt.title('Gesture Recognition Confusion Matrix')
plt.savefig("Results/conf_matrix.png")
plt.show()

print("\n--- Accuracy per Gesture ---")
accuracies = cm.diagonal() / cm.sum(axis=1)
for i, name in enumerate(class_labels):
    print(f"{name:<20}: {accuracies[i]:.2%}")

print(classification_report(y_true, y_pred, target_names=class_labels))

model_cnn.save('gesture_recognition_model.keras')