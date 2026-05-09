import cv2
import numpy as np
import os

DATA_DIR = "data/fire_images"
IMG_SIZE = 128
NUM_SAMPLES = 50

def generate_data():
    print("Generating synthetic data...")
    
    # ensure directories exist
    os.makedirs(f"{DATA_DIR}/fire", exist_ok=True)
    os.makedirs(f"{DATA_DIR}/no_fire", exist_ok=True)

    # Generate Fire images (Red/Orange dominanace)
    for i in range(NUM_SAMPLES):
        img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        # Random noise
        noise = np.random.randint(0, 100, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        # Add Red/Orange blobs
        cv2.circle(img, (np.random.randint(20, 100), np.random.randint(20, 100)), np.random.randint(10, 50), (0, 165, 255), -1) # Orange
        cv2.circle(img, (np.random.randint(20, 100), np.random.randint(20, 100)), np.random.randint(10, 40), (0, 0, 255), -1)   # Red
        
        img = cv2.add(img, noise)
        cv2.imwrite(f"{DATA_DIR}/fire/img_{i}.jpg", img)

    # Generate No Fire images (Green/Blue dominance - Forest/Sky)
    for i in range(NUM_SAMPLES):
        img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        # Green background
        img[:] = (0, 100, 0)
        # Random noise
        noise = np.random.randint(0, 50, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        # Add Blue (Sky) or Brown (Ground)
        cv2.rectangle(img, (0, 0), (IMG_SIZE, 50), (255, 0, 0), -1) # Sky
        
        img = cv2.add(img, noise)
        cv2.imwrite(f"{DATA_DIR}/no_fire/img_{i}.jpg", img)

    print(f"Generated {NUM_SAMPLES} images per class in {DATA_DIR}")

if __name__ == "__main__":
    generate_data()
