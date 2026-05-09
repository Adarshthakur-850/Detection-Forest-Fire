
# 🔥 Detection Forest Fire

An AI-powered **Forest Fire Detection System** that uses **Machine Learning / Deep Learning and Computer Vision** techniques to detect forest fires from images/video streams and help in early wildfire prevention.

## 📌 Overview

Forest fires cause massive environmental damage, wildlife loss, and economic destruction. Early detection is critical to reduce response time.

This project detects fire presence using image-based analysis and can be extended for:

- Real-time surveillance monitoring
- Drone-based forest monitoring
- CCTV wildfire detection
- Emergency alert systems
- Smart environmental monitoring systems

---

## 🚀 Features

- Detects fire from uploaded images
- Supports real-time webcam/video detection
- Uses Computer Vision for image preprocessing
- Deep Learning model for classification/detection
- Displays prediction results
- Can be integrated with alert systems

---

## 🛠️ Tech Stack

- Python
- OpenCV
- TensorFlow / Keras *(or PyTorch if used)*
- NumPy
- Pandas
- Matplotlib
- Flask / Streamlit *(if deployment UI exists)*

---

## 📂 Project Structure

```bash
Detection-Forest-Fire/
│
├── dataset/              # Training dataset
├── model/                # Saved trained model
├── notebooks/            # Jupyter notebooks
├── app.py                # Main application
├── train.py              # Model training script
├── predict.py            # Prediction script
├── requirements.txt      # Dependencies
└── README.md
````

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Adarshthakur-850/Detection-Forest-Fire.git
cd Detection-Forest-Fire
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset

The model is trained on fire/non-fire image datasets.

Dataset may include:

* Forest fire images
* Normal forest images
* Smoke/fire scenarios

You can use datasets from:

* Kaggle
* Roboflow
* Custom collected datasets

Example sources for similar datasets: ([GitHub][1])

---

## ▶️ Run the Project

### Train Model

```bash
python train.py
```

### Run Prediction

```bash
python predict.py
```

### Run Web App

```bash
python app.py
```

---

## Workflow

1. Collect dataset
2. Preprocess images
3. Train model
4. Evaluate accuracy
5. Perform real-time detection
6. Generate alerts/results

---

## Results

* Detects fire with high accuracy
* Works on image classification
* Can be extended to real-time wildfire monitoring systems

Example output:

✅ Fire Detected
❌ No Fire Detected

---

## Future Improvements

* Integrate drone surveillance
* Add IoT sensors
* SMS/Email emergency alerts
* Deploy on cloud
* Improve detection accuracy with YOLO models

---

## Applications

* Forest departments
* Disaster management agencies
* Environmental monitoring
* Wildlife protection
* Smart city safety systems

---

## Author

**Adarsh Thakur**
GitHub: [Adarshthakur-850](https://github.com/Adarshthakur-850?utm_source=chatgpt.com)

