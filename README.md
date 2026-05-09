# Forest Fire Detection System

A production-quality AI system that detects fire in images using a Convolutional Neural Network (CNN).

## Features
- **CNN Classification**: Trained model to distinguish between Fire and No Fire images.
- **Real-time Detection**: `detect_fire.py` logic for identifying fire regions.
- **FastAPI Backend**: REST API for image prediction.
- **Streamlit UI**: User-friendly web interface for uploading images.

## Project Structure
```
forest_fire_ai/
│
├── data/                # Synthetic training data
├── preprocessing/       # Data loading & processing
├── training/            # CNN training script
├── models/              # Saved .h5 model
├── detection/           # Inference logic
├── api/                 # FastAPI backend
├── ui/                  # Streamlit frontend
└── requirements.txt
```

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Generate Data** (required for first run):
    ```bash
    python data/generate_data.py
    ```

3.  **Train the Model**:
    ```bash
    python training/train_cnn.py
    ```

## Running the Application

### 1. Start the API (Backend)
```bash
uvicorn api.app:app --reload
```

### 2. Start the UI (Frontend)
Open a new terminal and run:
```bash
streamlit run ui/streamlit_app.py
```

## API Usage
**POST** `/predict` (Multipart/Form-Data)
- **file**: Image file
**Response**:
```json
{
  "label": "Fire",
  "confidence": 0.98
}
```
