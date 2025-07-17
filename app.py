
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Set page config
st.set_page_config(page_title="🧠 Brain Tumor Classifier", layout="centered")
st.title("🧠 Brain Tumor MRI Classifier")
st.markdown("Upload an MRI image and select the model to predict tumor type.")

# Class labels
class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Model selection
model_option = st.selectbox("Choose Model", ["Custom CNN", "MobileNetV2"])

# Load selected model
@st.cache_resource
def load_model(model_name):
    if model_name == "Custom CNN":
        return tf.keras.models.load_model("best_model.keras")
    else:
        return tf.keras.models.load_model("mobilenetv2_model.keras")

model = load_model(model_option)

# Image preprocessing
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# File uploader
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI", use_container_width=True)

    if st.button("🔍 Predict Tumor Type"):
        with st.spinner("Predicting..."):
            processed_img = preprocess_image(image)
            prediction = model.predict(processed_img)
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction) * 100

        st.success(f"🧠 Predicted Tumor Type: **{predicted_class}**")
        st.info(f"Confidence: {confidence:.2f}%")
