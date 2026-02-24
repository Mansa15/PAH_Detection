"""
PAH Detection - Streamlit Web Application
Run: streamlit run src/app.py
"""

import os
import json
import tempfile
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ─── CONFIG ───────────────────────────────────────────────────────────────────

IMG_SIZE   = (224, 224)
MODEL_PATH = "models/best_model.h5"
CLASS_IDX  = "models/class_indices.json"
THRESHOLD  = 0.5

# ─── LOAD MODEL (cached) ──────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_IDX) as f:
        class_indices = json.load(f)
    idx_to_label = {v: k for k, v in class_indices.items()}
    return model, idx_to_label


def preprocess(img: Image.Image) -> np.ndarray:
    img = img.resize(IMG_SIZE).convert("RGB")
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)


def get_gradcam(model, img_array, last_conv_layer_name="top_conv"):
    """Generate Grad-CAM heatmap for explainability."""
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads       = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap      = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap      = tf.squeeze(heatmap)
    heatmap      = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(orig_img: Image.Image, heatmap: np.ndarray, alpha=0.4):
    """Overlay Grad-CAM on original image."""
    heatmap_resized = np.array(
        Image.fromarray(np.uint8(255 * heatmap)).resize(orig_img.size, Image.LANCZOS)
    )
    colormap = cm.jet(heatmap_resized / 255.0)[:, :, :3]
    colormap = (colormap * 255).astype(np.uint8)
    orig_arr = np.array(orig_img.convert("RGB"))
    overlay  = (orig_arr * (1 - alpha) + colormap * alpha).astype(np.uint8)
    return Image.fromarray(overlay)


# ─── UI ───────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="PAH Detection System",
        page_icon="🫁",
        layout="wide",
    )

    # Header
    st.title("🫁 Pulmonary Artery Hypertension (PAH) Detection")
    st.markdown(
        "Upload a **chest X-ray or CT scan image** and the AI model will predict "
        "whether signs of Pulmonary Artery Hypertension are present."
    )
    st.divider()

    # Sidebar
    with st.sidebar:
        st.header("⚙ Settings")
        threshold = st.slider("Decision Threshold", 0.3, 0.9, THRESHOLD, 0.05,
                              help="Probability threshold for PAH classification")
        show_gradcam = st.checkbox("Show Grad-CAM (Explainability)", value=True)
        st.divider()
        st.info("**About PAH**\n\nPulmonary Arterial Hypertension is high blood pressure "
                "in the arteries of the lungs. Early detection significantly improves outcomes.")

    # File upload
    uploaded = st.file_uploader(
        "Upload Chest X-Ray / CT Scan",
        type=["jpg", "jpeg", "png", "bmp"],
        help="DICOM files should be pre-converted to JPEG/PNG",
    )

    if uploaded is None:
        st.markdown(
            "<div style='text-align:center; padding:60px; color:#888;'>"
            "⬆ Upload an image to begin analysis</div>",
            unsafe_allow_html=True,
        )
        return

    # Load model
    try:
        model, idx_to_label = load_model()
    except Exception as e:
        st.error(f"Could not load model: {e}\n\nPlease run `python src/train.py` first.")
        return

    # Display + predict
    img = Image.open(uploaded)
    arr = preprocess(img)

    with st.spinner("Analysing..."):
        prob = float(model.predict(arr, verbose=0)[0][0])

    label = idx_to_label.get(int(prob >= threshold), "Unknown")
    confidence = prob if prob >= threshold else 1 - prob
    risk = (
        "🟢 Low"        if prob < 0.3 else
        "🟡 Moderate"   if prob < 0.6 else
        "🔴 High"       if prob < 0.8 else
        "🚨 Very High"
    )

    # Layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input Image")
        st.image(img, use_container_width=True)

    with col2:
        st.subheader("Prediction Result")
        if label.upper() == "PAH":
            st.error(f"⚠ **PAH Detected**")
        else:
            st.success(f"✅ **Normal**")

        st.metric("PAH Probability", f"{prob:.2%}")
        st.metric("Confidence",       f"{confidence*100:.1f}%")
        st.metric("Risk Level",        risk)

        # Gauge chart (simple)
        fig, ax = plt.subplots(figsize=(4, 0.5))
        ax.barh(0, prob, color="#e74c3c" if prob >= threshold else "#2ecc71", height=0.4)
        ax.barh(0, 1 - prob, left=prob, color="#ecf0f1", height=0.4)
        ax.axvline(threshold, color="orange", linestyle="--", linewidth=1.5, label=f"Threshold={threshold}")
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel("Probability →")
        ax.legend(fontsize=7, loc="upper right")
        st.pyplot(fig, use_container_width=True)

    # Grad-CAM
    if show_gradcam:
        st.divider()
        st.subheader("🔍 Grad-CAM Explainability")
        st.caption("Highlighted regions show areas the model focused on for its prediction.")
        try:
            heatmap   = get_gradcam(model, arr)
            overlay   = overlay_heatmap(img, heatmap)
            c1, c2    = st.columns(2)
            c1.image(img,     caption="Original", use_container_width=True)
            c2.image(overlay, caption="Grad-CAM Overlay", use_container_width=True)
        except Exception as e:
            st.warning(f"Grad-CAM unavailable for this model architecture: {e}")

    # Disclaimer
    st.divider()
    st.caption(
        "⚠ **Medical Disclaimer**: This tool is for educational and research purposes only. "
        "It is NOT a substitute for professional medical diagnosis. Always consult a qualified physician."
    )


if __name__ == "__main__":
    main()
