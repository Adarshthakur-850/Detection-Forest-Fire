import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

IMG_SIZE = 64

def load_data(data_dir):
    images = []
    labels = []
    
    classes = ["no_fire", "fire"] # 0: no_fire, 1: fire
    
    for idx, label in enumerate(classes):
        path = os.path.join(data_dir, label)
        for img_name in os.listdir(path):
            try:
                img_path = os.path.join(path, img_name)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                images.append(img)
                labels.append(idx)
            except Exception as e:
                print(f"Error loading {img_name}: {e}")

    X = np.array(images).astype('float32') / 255.0
    y = to_categorical(np.array(labels), num_classes=2)
    
    return train_test_split(X, y, test_size=0.2, random_state=42)
