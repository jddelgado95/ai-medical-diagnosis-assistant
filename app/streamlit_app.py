import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import io
import os
from src.model import build_model
from src.utils import load_image, load_checkpoint
from src.explain import grad_cam, overlay_heatmap

st.set_page_config(page_title="AI Medical Assistant", layout="wide")

# Load model and weights
@st.cache_resource
def load_trained_model(weights_path, num_classes):
    model = build_model(num_classes=num_classes, pretrained=False)
    model = load_checkpoint(model, weights_path)
    return model

st.title("🧠 AI Medical Diagnosis Assistant")
st.markdown("Upload a medical image to classify and visualize the model's attention.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Load image for model
    input_tensor = load_image(uploaded_file)

    # Load model (assume 2 classes: Normal vs Abnormal)
    model = load_trained_model("weights/model.pth", num_classes=2)

    # Make prediction
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()

    class_names = ["Normal", "Abnormal"]
    st.markdown(f"### 🩺 Prediction: **{class_names[pred_idx]}** ({confidence*100:.2f}% confidence)")

    # Grad-CAM heatmap
    target_layer = model.layer4[-1]
    heatmap = grad_cam(model, input_tensor, target_layer)

    # Overlay heatmap
    image_np = np.array(image)
    heatmap_img = overlay_heatmap(image_np, heatmap)
    st.image(heatmap_img, caption="Grad-CAM Heatmap", use_column_width=True)