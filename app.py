import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="YOLO Object Detection", layout="centered")

st.title("YOLO Object Detection App")

# Load YOLO model
model = YOLO("yolo11n.pt")

uploaded_file = st.file_uploader(
    "Upload an image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Detect Objects"):
        with st.spinner("Detecting objects..."):
            results = model(np.array(image))
            result_img = results[0].plot()

        st.image(result_img, caption="Detected Objects", use_container_width=True)
