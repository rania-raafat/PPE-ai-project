import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===== CONFIG =====
st.set_page_config(page_title="PPE Safety AI System", layout="wide")

# ===== MODEL =====
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ===== STYLE =====
st.markdown("""
<style>
    .title { text-align:center; font-size:34px; font-weight:600; }
    .subtitle { text-align:center; color:gray; margin-bottom:20px; }
</style>
""", unsafe_allow_html=True)

# ===== HEADER =====
st.markdown("<div class='title'>PPE Safety Detection System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-powered compliance monitoring for industrial safety</div>", unsafe_allow_html=True)
st.markdown("---")

# ===== SIDEBAR =====
st.sidebar.header("Control Panel")
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.4)
mode = st.sidebar.radio("Input Mode", ["Upload Image", "Webcam"])

# ===== CLASS NAMES =====
CLASS_NAMES = {
    0: "Hardhat",
    1: "Mask",
    2: "NO-Hardhat",
    3: "NO-Mask",
    4: "NO-Safety Vest",
    5: "Person",
    6: "Safety Cone",
    7: "Safety Vest",
    8: "machinery",
    9: "vehicle"
}

PPE_CLASSES = {0, 1, 7}
VIOLATION_CLASSES = {2, 3, 4}

# ===== PREDICT (FIXED) =====
def predict(img):
    return model.predict(
        img,
        conf=confidence,
        iou=0.35,        # 🔥 stronger duplicate removal
        max_det=50,
        agnostic_nms=True
    )

# ===== INPUT =====
image = None

if mode == "Upload Image":
    file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if file:
        image = Image.open(file).convert("RGB")

elif mode == "Webcam":
    cam = st.camera_input("Capture Image")
    if cam:
        image = Image.open(cam).convert("RGB")

# ===== MAIN =====
if image is not None:

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Detection Output")
        results = predict(image)
        st.image(results[0].plot(), use_container_width=True)

    # ===== REMOVE DUPLICATES (IMPORTANT FIX) =====
    detections = []
    seen = []

    def is_same_object(box1, box2):
        x11, y11, x12, y12 = box1
        x21, y21, x22, y22 = box2

        xi1 = max(x11, x21)
        yi1 = max(y11, y21)
        xi2 = min(x12, x22)
        yi2 = min(y12, y22)

        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        area1 = (x12 - x11) * (y12 - y11)
        area2 = (x22 - x21) * (y22 - y21)

        union = area1 + area2 - inter
        if union == 0:
            return False

        return (inter / union) > 0.5

    for r in results:
        for box in r.boxes:

            cls_id = int(box.cls)
            conf = float(box.conf)

            # filter weak detections
            if conf < confidence:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            current_box = (x1, y1, x2, y2)

            duplicate = False
            for old_box, old_cls in seen:
                if old_cls == cls_id and is_same_object(current_box, old_box):
                    duplicate = True
                    break

            if duplicate:
                continue

            seen.append((current_box, cls_id))

            detections.append({
                "class_id": cls_id,
                "label": CLASS_NAMES.get(cls_id, f"class_{cls_id}"),
                "confidence": round(conf, 3),
                "category": (
                    "✅ PPE Worn" if cls_id in PPE_CLASSES else
                    "⚠️ Violation" if cls_id in VIOLATION_CLASSES else
                    "ℹ️ Other"
                )
            })

    df = pd.DataFrame(detections)

    counts = df["class_id"].value_counts().to_dict() if not df.empty else {}

    hardhat = counts.get(0, 0)
    mask = counts.get(1, 0)
    no_hardhat = counts.get(2, 0)
    no_mask = counts.get(3, 0)
    no_vest = counts.get(4, 0)
    person = counts.get(5, 0)
    safety_cone = counts.get(6, 0)
    safety_vest = counts.get(7, 0)
    machinery = counts.get(8, 0)
    vehicle = counts.get(9, 0)

    total_violations = no_hardhat + no_mask + no_vest

    # ===== UI =====
    st.markdown("### 🦺 PPE Status")
    c1, c2, c3 = st.columns(3)
    c1.metric("Hardhats ✅", hardhat)
    c2.metric("Masks ✅", mask)
    c3.metric("Safety Vests ✅", safety_vest)

    st.markdown("### 🚨 Violations")
    v1, v2, v3 = st.columns(3)
    v1.metric("NO-Hardhat ⚠️", no_hardhat)
    v2.metric("NO-Mask ⚠️", no_mask)
    v3.metric("NO-Safety Vest ⚠️", no_vest)

    st.markdown("### 🏗️ Scene Objects")
    o1, o2, o3, o4 = st.columns(4)
    o1.metric("Persons 👤", person)
    o2.metric("Safety Cones 🔺", safety_cone)
    o3.metric("Machinery 🏗️", machinery)
    o4.metric("Vehicles 🚗", vehicle)

    # ===== STATUS =====
    if total_violations > 0:
        st.error("🚨 Non-Compliant detected")
    else:
        st.success("✅ Safe Scene Detected")

    # ===== REPORT =====
    if total_violations > 0:
        st.markdown("### Detection Report")
        st.dataframe(df)

# ===== FOOTER =====
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:gray;'>AI Safety Monitoring System • Made by Enas & Rania</div>",
    unsafe_allow_html=True
)