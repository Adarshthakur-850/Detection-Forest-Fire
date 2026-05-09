import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing.preprocess import load_data

DATA_DIR = "data/fire_images"
MODEL_PATH = "models/fire_model.h5"

def train_model():
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data(DATA_DIR)
    
    # Check if data loaded correctly
    if len(X_train) == 0:
        print("No data found! Please run data/generate_data.py first.")
        return

    print(f"Training on {len(X_train)} samples, Validating on {len(X_test)} samples.")

    # Build CNN
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax') # 2 classes: Fire, No Fire
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print("Training model...")
    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)

    # Save Model
    os.makedirs("models", exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # Plot results (saved to file instead of plt.show() for headless environments)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.legend()
    plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Loss')
    
    plt.savefig('training/training_plot.png')
    print("Training plot saved to training/training_plot.png")

if __name__ == "__main__":
    train_model()
