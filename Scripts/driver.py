import requests
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import keras

# Libraries for collecting data with Mindrove EMG Sensor ########################
import mindrove
from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds
import mindrove.data_filter as MindroveFilter
import time
import scipy
from scipy import signal
from enum import Enum
import csv
import math
import matplotlib.pyplot as plt
RMSTHRESHOLD=3.0
##################################################################################


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
wait = False

# Load gesture prediction model
gesture_model = load_model(r"C:\Users\gianc\SeniorDesign\gesture_recognition_model.keras")

#window_size = 125 ##Initialized to 2 seconds in Mindrove Initialization Below
channels = 8
gesture_commands = {
    'Swipe':'L-Sign',
    'A-F':'Middle Pinch',
    'G-M':'Ring Pinch',
    'N-T':'Pinky Pinch',
    'U-Z':'Three Fingers',
    'Delete':'Thumb-Out',
    'Space/Enter':'Knock',
    'Start/Stop':'Surfs Up'
}

gesture_strings = [
    'Rest',
    'Middle Pinch',
    'Ring Pinch',
    'Pinky Pinch',
    'L-Sign',
    'Thumb-Out',
    'Knock',
    'Three Fingers',
    'Surfs Up'
]
##buffer = np.zeros((window_size, channels))

##################################################################

# Iitializing Mindrove EMG Sensor #
BoardShim.enable_board_logger() # enable logger when developing to catch relevant logs
params = MindRoveInputParams()
board_id = BoardIds.MINDROVE_WIFI_BOARD
board_shim = BoardShim(board_id, params)

board_shim.prepare_session()

emg_channels = BoardShim.get_emg_channels(board_id)
#accel_channels = BoardShim.get_accel_channels(board_id)
sampling_rate = BoardShim.get_sampling_rate(board_id)

window_size = 2 # seconds
window_size_samples = 125  # Match training window_size
num_points = window_size_samples*8
print()
OffSet=300

def reset():
    global left, right, taps, word_id, run, typing, checking
    left = ""
    right = " "
    taps = []
    word_id = 0

    run = False
    typing = False
    checking = False

# Get live EMG data
def get_emg_data(num_points):
    print("Getting EMG Data...")
    #blocking loop
    while board_shim.get_board_data_count() < num_points:
        time.sleep(0.01)
    data = board_shim.get_current_board_data(num_points) # Note that using this command, there will be an overlap in the data obtained in the current cycle and the previous cycle(s)
    emg_data = data[emg_channels] # output of shape (8, num_of_samples) ## Beware that depending on the electrode configuration, some channels can be *inactive*, resulting in all-zero data for that particular channel
    # process data, or print it out
    return emg_data

# Preprocess data before passing to gesture prediction model (normalize/ filter)
def process_data(EMGData):
    print("Processing Data...")
    b, a = signal.butter(
        4,
        [20, 80],
        btype='bandpass',
        fs=sampling_rate
    )

    filtered = signal.filtfilt(b, a, EMGData, axis=1)
   
    b_notch, a_notch = signal.iirnotch(
        60,
        30,
        fs=sampling_rate
    )

    filtered = signal.filtfilt(b_notch, a_notch, filtered, axis=1)

    return filtered

#Gesture Prediction Voting Setup
CUSTOM_THRESHOLDS = {
    1:  0.55, 
    3:  0.40, # Middle Pinch
    4:  0.40, # Ring Pinch
    9:  0.65, # Pinky Up
    11: 0.35, 
}

EMA_DECAY = 0.3          # How much weight new predictions carry vs history
PROB_TEMP = 0.8
ENTER_THRESHOLD = 0.5   # Confidence needed to switch to a new gesture
EXIT_THRESHOLD = 0.2    # Confidence drop needed to revert to Rest
NUM_CLASSES = len(gesture_strings)
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

# Call the FlexType model
def flexType(taps):
    taps_payload = {
        "keyboardId": keyboard_id,
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


board_shim.start_stream()
print("Starting Predictions...")

voter = GestureVoter(NUM_CLASSES, decay=EMA_DECAY, thresholds=CUSTOM_THRESHOLDS)
lastGesture = ""

try:
    while True:
####### Gesture Prediction Logic ##########
        EMGData = get_emg_data(num_points)
        processedData = process_data(EMGData)

        processedData = processedData.T
        processedData = np.expand_dims(processedData, axis=0)
        gesture_pred = gesture_model.predict(processedData)

        gesture_idx = voter.predict_voted_gesture(gesture_pred[0], temperature=PROB_TEMP)
        gesture_label = gesture_strings[gesture_idx]

        confidence = voter.ema_probs[gesture_idx]

        if gesture_label != lastGesture:
            if(confidence > .8):
                print(f"Predicted: {gesture_label} with confidence {confidence}")
            else:
                print(f"Low confidence ({confidence}) for predicted gesture: {gesture_label}")

        lastGesture = gesture_label
        
####### Word Prediction Logic ##########
        if (run == False and gesture_label == "Surfs Up"):
            run = True
            typing = True
            checking = False
            wait = True

        elif run:
            if gesture_label in gesture_commands:
                if gesture_label == "Surfs Up" and not wait:
                    wait = True
                    reset() # Reset typing
                    continue

                if wait and gesture_label == "Rest":
                    wait = False
                    continue

                if typing:
                    if not wait:
                        match gesture_label:
                            case 'A-F':
                                taps.append({"touches": [{"x": 0, "y": 0}], "certain": True})
                                wait = True
                            case 'G-M':
                                taps.append({"touches": [{"x": 100, "y": 0}], "certain": True})
                                wait = True
                            case 'N-T':
                                taps.append({"touches": [{"x": 200, "y": 0}], "certain": True})
                                wait = True
                            case 'U-Z':
                                taps.append({"touches": [{"x": 300, "y": 0}], "certain": True})
                                wait = True
                            case 'Space/Enter': # Compile and check word
                                wait = True
                                typing = False
                                checking = True
                                word_pred = flexType(taps)

                                print("\nPredicted Word: ")
                                if "best" in word_pred and word_pred["best"]:
                                    for i, pred in enumerate(word_pred["best"]):
                                        print(f"Option {i}: {pred['text']}")
                                else:
                                    print("No words predicted.")
                if checking:
                    if not wait:
                        if "best" in word_pred and len(word_pred["best"]) > 0:
                            word = word_pred["best"][word_id]["text"]
                            print(f"CURRENT SELECTION: [{word}] (Gesture Swipe to change)")
                        match gesture_label:
                            case 'Swipe': # Check next guessed word
                                wait = True
                                word_id+=1
                                if word_id >= len(word_pred.get("best", [])):
                                    word_id = 0
                                word = word_pred["best"][word_id]["text"]
                                print(f"Swiped! New Selected Word: [{word}]")
                            case 'Delete': # Delete just-typed word
                                wait = True
                                typing = True
                                checking = False
                                word_id = 0
                                taps = []
                            case 'Space/Enter': # Continue to next word
                                wait = True
                                typing = True
                                checking = False
                                left += word + " "
                                word_id = 0
                                print(f"\nFull Message:\n {left}")
                                taps = []

            time.sleep(0.1)

except KeyboardInterrupt:
    print("Stopping Predictions...")
    board_shim.stop_stream()
    board_shim.release_session()

#     # buffer = np.roll(buffer, -1, axis=0)
#     # buffer[-1, :] = raw_data