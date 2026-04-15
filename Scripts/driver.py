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

# Load gesture prediction model
gesture_model = load_model(r"C:\Users\gianc\SeniorDesign\gesture_recognition_model.keras")

#window_size = 125 ##Initialized to 2 seconds in Mindrove Initialization Below
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

gesture_strings = [
    'Rest',
    'Two-Finger Wave',
    'Middle Pinch',
    'Ring Pinch',
    'Pinky Pinch',
    'L-Sign',
    'Thumb-Out',
    'Knock',
    'Wiggle Fingers'
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

# def WriteToCSV(EMGData,CSVWriter):

#     DataToWrite=process_data(EMGData)
#     for i in range (DataToWrite.shape[1]):
#         row=[DataToWrite[j,i] for j in range (8)]
#         CSVWriter.writerow(row)


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


board_shim.start_stream()
print("Starting Predictions...")

try:
    while True:
        EMGData = get_emg_data(num_points)
        processedData = process_data(EMGData)

        processedData = processedData.T
        processedData = np.expand_dims(processedData, axis=0)
        gesture_pred = gesture_model.predict(processedData)
        confidence = np.max(gesture_pred)
        gesture_label = gesture_strings[np.argmax(gesture_pred)]

        if(confidence > .8):
            print(f"Predicted: {gesture_label} with confidence {confidence}")
        else:
            print(f"Low confidence ({confidence}) for predicted gesture: {gesture_label}")

        time.sleep(3)
except KeyboardInterrupt:
    print("Stopping Predictions...")
    board_shim.stop_stream()
    board_shim.release_session()
   

# try:
#     board_shim.stop_stream()
# except:
#     pass
# #board_shim.start_stream()

# # Live data loop
# while True:

#     # raw_data = get_emg_data(num_points) # NEED TO BUILD FUNCTION

#     # buffer = np.roll(buffer, -1, axis=0)
#     # buffer[-1, :] = raw_data

#     # input_emg = process_data(EMGData) # NEED TO BUILD FUNCTION TO NORMALIZE/ FILTER DATA
   
#     gesture_pred = model.predict(input_emg)
#     gesture_label = class_names[np.argmax(gesture_pred)]

#     # print(f"Predicted: {gestures[gesture_id]}")

#     if (run == False and gesture_label == "Start/Stop"):
#         run = True
#         typing = True

#     if run:
#         if gesture_label in gesture_commands:
#             if gesture_label == "Start/Stop":
#                 reset() # Reset typing
#             if typing:
#                 match gesture_label:
#                     case 'A-F':
#                         taps.append({"touches": [{"x": 0, "y": 0}], "certain": True})
#                     case 'G-M':
#                         taps.append({"touches": [{"x": 100, "y": 0}], "certain": True})
#                     case 'N-T':
#                         taps.append({"touches": [{"x": 200, "y": 0}], "certain": True})
#                     case 'U-Z':
#                         taps.append({"touches": [{"x": 300, "y": 0}], "certain": True})
#                     case 'Space/Enter': # Compile and check word
#                         typing = False
#                         checking = True
#                         word_pred = flexType(taps)
#             if checking:
#                 word = word_pred["best"][word_id]["text"]
#                 match gesture_label:
#                     case 'Swipe': # Check next guessed word
#                         word_id+=1
#                         if word_id >= len(word_pred["best"]):
#                             word_id = 0
#                     case 'Delete': # Delete just-typed word
#                         typing = True
#                         checking = False
#                         word_id = 0
#                     case 'Space/Enter': # Continue to next word
#                         typing = True
#                         checking = False
#                         left += word + " "
#                         word_id = 0