import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
from PIL import Image
import tempfile
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="ANPR System",
    layout="centered"
)

st.title("🚗 Automatic Number Plate Recognition (ANPR)")
st.write("Upload an image or video to detect and read number plates")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "best.pt")
    return YOLO(model_path)

model = load_model()

# ---------------- LOAD OCR ----------------
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25)

mode = st.sidebar.radio(
    "Select Input Type",
    ["Image", "Video"]
)

# ==================================================
# IMAGE MODE
# ==================================================
if mode == "Image":
    uploaded_image = st.file_uploader(
        "📤 Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image:
        # ---------- SAFE IMAGE LOADING ----------
        image = Image.open(uploaded_image).convert("RGB")  # FIX: RGBA → RGB
        img = np.array(image)

        st.image(image, caption="Uploaded Image", use_container_width=True)

        with st.spinner("🔍 Detecting number plate..."):
            results = model(img, conf=conf_threshold)

        detected = False

        for box in results[0].boxes.xyxy:
            detected = True
            x1, y1, x2, y2 = map(int, box)
            plate = img[y1:y2, x1:x2]

            # OCR
            text = reader.readtext(plate, detail=0)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if text:
                cv2.putText(
                    img, text[0],
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2
                )
                st.success(f"✅ Detected Plate Number: {text[0]}")
            else:
                st.warning("⚠️ Plate detected but text not clear")

            st.image(plate, caption="Detected Plate", use_container_width=True)

        if not detected:
            st.error("❌ No number plate detected")

        st.image(img, caption="Final Output", use_container_width=True)

# ==================================================
# VIDEO MODE
# ==================================================
if mode == "Video":
    uploaded_video = st.file_uploader(
        "📤 Upload Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video:
        # ---------- SAVE TEMP VIDEO ----------
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        st.info("▶️ Processing video...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # FIX: Ensure 3 channels
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            results = model(frame, conf=conf_threshold)

            for box in results[0].boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                plate = frame[y1:y2, x1:x2]

                text = reader.readtext(plate, detail=0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if text:
                    cv2.putText(
                        frame, text[0],
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2
                    )

            stframe.image(frame, channels="BGR", use_container_width=True)
            time.sleep(0.03)  # control FPS (CPU friendly)

        cap.release()
        st.success("✅ Video processing completed")

