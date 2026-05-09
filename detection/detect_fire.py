import cv2
import numpy as np
import tensorflow as tf
import os

MODEL_PATH = "models/fire_model.h5"

class FireDetector:
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise IOError("Model not found. Please train the model first.")
        self.model = tf.keras.models.load_model(MODEL_PATH)
        self.classes = ["No Fire", "Fire"]

    def predict(self, frame):
        # Preprocess for CNN
        img = cv2.resize(frame, (64, 64))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0) # Add batch dimension

        prediction = self.model.predict(img)
        class_idx = np.argmax(prediction)
        confidence = float(prediction[0][class_idx])
        label = self.classes[class_idx]

        return label, confidence

    def detect_and_draw(self, frame):
        label, confidence = self.predict(frame)

        # Draw Label
        color = (0, 255, 0) if label == "No Fire" else (0, 0, 255)
        cv2.putText(frame, f"{label} ({confidence:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Draw Bounding Box (Heuristic: Color Thresholding)
        if label == "Fire":
            # Convert to HSV to find orange/red regions
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # Define lower/upper bounds for Fire color (Orange-Red)
            lower_fire = np.array([18, 50, 50], dtype="uint8")
            upper_fire = np.array([35, 255, 255], dtype="uint8")
            
            mask = cv2.inRange(hsv, lower_fire, upper_fire)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                if cv2.contourArea(cnt) > 500: # Filter small noise
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        return frame, label
