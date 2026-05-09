import streamlit as st
import requests
from PIL import Image
import numpy as np
import cv2

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Forest Fire Detector", page_icon="🔥")

st.title("🔥 Forest Fire Detection System")
st.markdown("Upload an image to detect if it contains fire.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button("Detect Fire"):
        with st.spinner("Analyzing..."):
            try:
                # Reset pointer
                uploaded_file.seek(0)
                files = {"file": uploaded_file.getvalue()}
                response = requests.post(API_URL, files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    label = result['label']
                    conf = result['confidence']
                    
                    if label == "Fire":
                        st.error(f"🚨 **Result: {label} Detected!**")
                        st.write(f"Confidence: {conf:.2%}")
                    else:
                        st.success(f"✅ **Result: {label}**")
                        st.write(f"Confidence: {conf:.2%}")
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Connection Error: {e}. Is the API running?")
