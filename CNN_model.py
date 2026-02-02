import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models

# Data configuration
DATA_PATH = "CPE4850 - Gesture Data/"  #
CHANNELS = [f'FilteredChannel{i}' for i in range(1, 9)]
NUM_GESTURES = 12
WINDOW_SIZE = 500


def load_data(base_path):
    X, y = [], []
    file_pattern = os.path.join(base_path, "**", "*.csv")
    files = glob.glob(file_pattern, recursive=True)

    for file in files:
        label = file.split(os.sep)[-3]

        df = pd.read_csv(file, sep='\t')
        data = df[CHANNELS].values

        if len(data) >= WINDOW_SIZE:
            data = data[:WINDOW_SIZE]
        else:
            data = np.pad(data, ((0, WINDOW_SIZE - len(data)), (0, 0)), mode='edge')

        X.append(data)
        y.append(label)

    return np.array(X), np.array(y)


# Pre-Process Dats
print("Loading data...")
X, y = load_data(DATA_PATH)

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Normalize features
scaler = StandardScaler()
X_shape = X.shape
X = scaler.fit_transform(X.reshape(-1, len(CHANNELS))).reshape(X_shape)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded)


# Define CNN model
def build_cnn_model(input_shape, num_classes):
    model = models.Sequential([

        # First block
        layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),

        # Second block
        layers.Conv1D(128, kernel_size=3, activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),  # Condenses temporal info

        # Fully connected layers
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


model = build_cnn_model((WINDOW_SIZE, 8), NUM_GESTURES)
model.summary()

# Train model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
)

# 5. Evaluation
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc * 100:.2f}%")