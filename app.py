import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

st.set_page_config(page_title="Road Damage Detection", layout="centered")

@st.cache_resource
def load_model():
    return YOLO("D:\\OneDrive\\Desktop\\Damage\\best.pt")  # Make sure best.pt is in same folder

model = load_model()

st.title("🚧 Road Damage Detection System")
st.write("Upload a road image to detect potholes, cracks, and manholes.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    suffix = os.path.splitext(uploaded_file.name)[1]

    # Safely read uploaded file
    uploaded_file.seek(0)
    file_bytes = uploaded_file.read()

    # Write to temp file with extension
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.write(file_bytes)
    temp_file.flush()
    temp_file.close()

    temp_path = temp_file.name

    # Run prediction
    results = model.predict(source=temp_path, conf=0.25)

    r = results[0]
    img = Image.fromarray(r.plot())
    st.image(img, caption="Detection Result", use_column_width=True)

    if len(r.boxes) == 0:
        st.warning("No damage detected.")
    else:
        img_h, img_w = r.orig_shape

        for box in r.boxes.xywh.cpu().numpy():
            _, _, w, h = box
            area_ratio = (w * h) / (img_w * img_h)

            if area_ratio < 0.02:
                severity = "Minor"
                cost = "₹2,000 – ₹5,000"
            elif area_ratio < 0.08:
                severity = "Moderate"
                cost = "₹5000 – ₹10,000"
            else:
                severity = "Severe"
                cost = "₹10,000+"

            st.markdown(f"### 🛠 Severity: **{severity}**")
            st.markdown(f"### 💰 Estimated Repair Cost: **{cost}**")

    # Do not delete temp file on Windows to avoid PermissionError