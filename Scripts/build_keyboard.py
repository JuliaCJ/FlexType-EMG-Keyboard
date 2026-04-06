import requests
import json

API_BASE = "https://api.imagineville.org"

keyboard_payload = {
        "keys": [
            {"labels": ["<sp>"], "x": -100, "y": -100, "width": 1, "height": 1},
            {"labels": ["a", "b", "c", "d", "e", "f"], "x": 0, "y": 0, "width": 1, "height": 1},
            {"labels": ["g", "h", "i", "j", "k", "l", "m"], "x": 100, "y": 0, "width": 1, "height": 1},
            {"labels": ["n", "o", "p", "q", "r", "s", "t"], "x": 200, "y": 0, "width": 1, "height": 1},
            {"labels": ["u", "v", "w", "x", "y", "z"], "x": 300, "y": 0, "width": 1, "height": 1}
            ],
        "lang": "en",
        "type": "STEP"
    }

resp = requests.post(f"{API_BASE}/keyboard/create", json=keyboard_payload)
print(resp.content)

keyboard = resp.json()
keyboard_id = keyboard["id"]

print("Keyboard created with ID: ", keyboard_id)
