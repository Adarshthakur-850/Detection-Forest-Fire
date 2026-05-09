import requests
import cv2
import numpy as np
import os
import time

# Ensure API is running before executing this script
API_URL = "http://127.0.0.1:8000/predict"
TEST_IMG_PATH = "data/fire_images/fire/img_0.jpg"

def test_api():
    print(f"Testing API with image: {TEST_IMG_PATH}")
    if not os.path.exists(TEST_IMG_PATH):
        print("Test image not found. generating data first...")
        import sys
        sys.path.append('.')
        from data import generate_data
        generate_data.generate_data()

    try:
        with open(TEST_IMG_PATH, "rb") as f:
            files = {"file": f}
            response = requests.post(API_URL, files=files)
            
            if response.status_code == 200:
                print("API Response:", response.json())
            else:
                print(f"API Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Failed to connect to API: {e}")

if __name__ == "__main__":
    test_api()
