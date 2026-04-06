import requests
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# API call to FlexType model
API_BASE = "https://api.imagineville.org"
keyboard_id = "dbca8af9683b49d197ae0ac73ddfb850"
left = ""
right = " "
taps = []
word_id = 0

# Declare states
run = False
typing = False
checking = False

# Load gesture prediction model
gesture_model = load_model(gesture_recognition_model.keras)

window_size = 3000
channels = 8
gesture_commands = {
    'Swipe':'Two-Finger Wave',
    'A-F':'Middle Pinch',
    'G-M':'Ring Pinch',
    'N-T':'Pinky Pinch',
    'U-Z':'L-Sign',
    'Delete':'Thumb-Out',
    'Space/Enter':'Knock',
    'Start/Stop':'Wiggle Fingers'
}

buffer = np.zeros((window_size, channels))

def reset():
    left = ""
    right = " "
    taps = []
    word_id = 0

    run = False
    typing = False
    checking = False

# Get live EMG data
def get_emg_data():
    print("Getting EMG Data...")

# Preprocess data before passing to gesture prediction model (normalize/ filter)
def process_data(buffer):
    print("Processing Data...")

# Call the FlexType model
def flexType(taps):
    taps_payload = {
        "keyboard": keyboard_id,
        "vocab": "100k",
        "sort": "logprob",
        "safe": True,
        "singleWord": True,
        "exactContext": True,
        "lang": "en",
        "config": "default",
        "numBest": 3,
        "numPrefix": 0,

        # Context
        "left": left,
        "right": right,

        # Tap coordinates
        "taps": taps
    }

    # Send request
    response = requests.post(f"{API_BASE}/rec/taps", json=taps_payload)

    # Return results
    word_pred = response.json()
    return word_pred

# Live data loop
while True:
    raw_data = get_emg_data() # NEED TO BUILD FUNCTION

    buffer = np.roll(buffer, -1, axis=0)
    buffer[-1, :] = raw_data

    input_emg = process_data(buffer) # NEED TO BUILD FUNCTION TO NORMALIZE/ FILTER DATA

    gesture_pred = model.predict(input_emg)
    gesture_label = class_names[np.argmax(gesture_pred)]

    print(f"Predicted: {gestures[gesture_id]}")

    if (run == False and gesture_label == "Start/Stop"):
        run = True
        typing = True

    if run:
        if gesture_label in gesture_commands:
            if gesture_label == "Start/Stop":
                reset() # Reset typing
            if typing:
                match gesture_label:
                    case 'A-F':
                        taps.append({"touches": [{"x": 0, "y": 0}], "certain": True})
                    case 'G-M':
                        taps.append({"touches": [{"x": 100, "y": 0}], "certain": True})
                    case 'N-T':
                        taps.append({"touches": [{"x": 200, "y": 0}], "certain": True})
                    case 'U-Z':
                        taps.append({"touches": [{"x": 300, "y": 0}], "certain": True})
                    case 'Space/Enter': # Compile and check word
                        typing = False
                        checking = True
                        word_pred = flexType(taps)
            if checking:
                word = word_pred["best"][word_id]["text"]
                match gesture_label:
                    case 'Swipe': # Check next guessed word
                        word_id+=1
                        if word_id >= len(word_pred["best"]):
                            word_id = 0
                    case 'Delete': # Delete just-typed word
                        typing = True
                        checking = False
                        word_id = 0
                    case 'Space/Enter': # Continue to next word
                        typing = True
                        checking = False
                        left.append(word + " ")
                        word_id = 0




