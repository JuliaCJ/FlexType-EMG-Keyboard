import requests
import json

API_BASE = "https://api.imagineville.org"

keyboard_id = "dbca8af9683b49d197ae0ac73ddfb850"

# Test keyboard
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
    "left": "",
    "right": "",

    # Tap coordinates
    "taps": [
        {"touches": [{"x":100, "y":0}], "certain":True}, # H
        {"touches": [{"x": 0, "y": 0}], "certain": True}, # E
        {"touches": [{"x": 100, "y": 0}], "certain": True}, # L
        {"touches": [{"x": 100, "y": 0}], "certain": True}, # L
        {"touches": [{"x": 200, "y": 0}], "certain": True} # O
    ]
}

# Send request
response = requests.post(f"{API_BASE}/rec/taps", json=taps_payload)

# Get the JSON data directly as a dictionary
word_pred = response.json()

# Iterate through the "best" predictions list
for item in word_pred.get("best", []):
    # Print the text value for each prediction
    print(item.get("text"))
