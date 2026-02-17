# detect_client.py
# Client script that sends image + API key to your Cancer Detector server

import requests

API_URL = "http://127.0.0.1:8000"   # change if server runs elsewhere
API_KEY = "password"                        # must match server.py



def detect_cancer(image_path: str):
    payload = {
        "image_path": image_path,
        "api_key": API_KEY
    }

    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()

        result = response.json()

        print("\n===== Prediction Result =====")
        print(f"Prediction: {result['prediction']}")
        print(f"Meaning: {result['label_meaning']}")

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    # change this to any image you wanna test
    detect_cancer("test_image.jpg")