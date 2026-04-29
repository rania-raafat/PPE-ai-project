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
    .card { padding:15px; border-radius:12px; background:#111827; border:1px solid #2d2d2d; }
    .kpi { padding:15px; border-radius:10px; background:#0f172a; text-align:center; border:1px solid #1f2937; }
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

# ===== CLASS NAMES — exactly as trained (10 classes) =====
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

# PPE worn correctly (compliant detections)
PPE_CLASSES = {0: "Hardhat", 1: "Mask", 7: "Safety Vest"}

# Violations (missing PPE)
VIOLATION_CLASSES = {2: "NO-Hardhat", 3: "NO-Mask", 4: "NO-Safety Vest"}

# Other objects (not PPE-related)
OTHER_CLASSES = {5: "Person", 6: "Safety Cone", 8: "machinery", 9: "vehicle"}

# ===== PREDICT =====
def predict(img):
    return model.predict(np.array(img), conf=confidence)

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
        output = results[0].plot()
        st.image(output, use_container_width=True)

    # ===== EXTRACT DATA =====
    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls)
            detections.append({
                "class_id": cls_id,
                "label": CLASS_NAMES.get(cls_id, f"class_{cls_id}"),
                "confidence": round(float(box.conf), 3),
                "category": (
                    "✅ PPE Worn" if cls_id in PPE_CLASSES else
                    "⚠️ Violation" if cls_id in VIOLATION_CLASSES else
                    "ℹ️ Other"
                )
            })

    df = pd.DataFrame(detections)

    # Count per class
    counts = df["class_id"].value_counts().to_dict() if not df.empty else {}

    hardhat      = counts.get(0, 0)
    mask         = counts.get(1, 0)
    no_hardhat   = counts.get(2, 0)
    no_mask      = counts.get(3, 0)
    no_vest      = counts.get(4, 0)
    person       = counts.get(5, 0)
    safety_cone  = counts.get(6, 0)
    safety_vest  = counts.get(7, 0)
    machinery    = counts.get(8, 0)
    vehicle      = counts.get(9, 0)
    total        = len(df)

    total_violations = no_hardhat + no_mask + no_vest

    # ===== KPIs — PPE =====
    st.markdown("### 🦺 PPE Status")
    c1, c2, c3 = st.columns(3)
    c1.metric("Hardhats ✅", hardhat)
    c2.metric("Masks ✅", mask)
    c3.metric("Safety Vests ✅", safety_vest)

    st.markdown("### 🚨 Violations")
    v1, v2, v3 = st.columns(3)
    v1.metric("NO-Hardhat ⚠️", no_hardhat,
              delta=f"-{no_hardhat}" if no_hardhat else None, delta_color="inverse")
    v2.metric("NO-Mask ⚠️", no_mask,
              delta=f"-{no_mask}" if no_mask else None, delta_color="inverse")
    v3.metric("NO-Safety Vest ⚠️", no_vest,
              delta=f"-{no_vest}" if no_vest else None, delta_color="inverse")

    st.markdown("### 🏗️ Scene Objects")
    o1, o2, o3, o4 = st.columns(4)
    o1.metric("Persons 👤", person)
    o2.metric("Safety Cones 🔺", safety_cone)
    o3.metric("Machinery 🏗️", machinery)
    o4.metric("Vehicles 🚗", vehicle)

    # ===== SAFETY COMPLIANCE LOGIC =====
    st.markdown("### Safety Compliance Status")

    violations_list = []
    if no_hardhat > 0: violations_list.append(f"{no_hardhat} person(s) without hardhat")
    if no_mask    > 0: violations_list.append(f"{no_mask} person(s) without mask")
    if no_vest    > 0: violations_list.append(f"{no_vest} person(s) without safety vest")

    if total_violations > 0:
        st.error(f"🚨 Non-Compliant: " + " | ".join(violations_list))
    elif hardhat == 0 and mask == 0 and safety_vest == 0:
        st.info("ℹ️ No PPE items detected in this image.")
    else:
        ppe_summary = []
        if hardhat > 0:     ppe_summary.append(f"{hardhat} Hardhat(s)")
        if mask > 0:        ppe_summary.append(f"{mask} Mask(s)")
        if safety_vest > 0: ppe_summary.append(f"{safety_vest} Safety Vest(s)")
        st.success(f"✅ Full Compliance: {', '.join(ppe_summary)} detected correctly.")

    # ===== CHART =====
    if total > 0:
        st.markdown("### Detection Distribution")
        present_classes = {k: v for k, v in CLASS_NAMES.items() if counts.get(k, 0) > 0}
        labels = list(present_classes.values())
        values = [counts[k] for k in present_classes]

        color_map = {
            0: "#22c55e",   # Hardhat — green
            1: "#3b82f6",   # Mask — blue
            2: "#ef4444",   # NO-Hardhat — red
            3: "#f97316",   # NO-Mask — orange
            4: "#eab308",   # NO-Safety Vest — yellow
            5: "#94a3b8",   # Person — gray
            6: "#06b6d4",   # Safety Cone — cyan
            7: "#10b981",   # Safety Vest — emerald
            8: "#8b5cf6",   # machinery — purple
            9: "#f59e0b",   # vehicle — amber
        }
        colors = [color_map[k] for k in present_classes]

        fig, ax = plt.subplots()
        ax.pie(values, labels=labels, autopct="%1.1f%%", colors=colors)
        ax.set_title("Detected Classes")
        st.pyplot(fig)

    # ===== REPORT =====
    st.markdown("### Detection Report")
    if df.empty:
        st.info("No detections found at the current confidence threshold. Try lowering the threshold.")
    else:
        st.dataframe(df[["label", "category", "confidence"]], use_container_width=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Report", csv, "ppe_report.csv", "text/csv")

# ===== FOOTER =====
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:gray;'>AI Safety Monitoring System • Made by Enas & Rania</div>",
    unsafe_allow_html=True
)