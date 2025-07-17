# Project Name - Brain Tumor MRI Image Classification.

# STEP 1 - Upload and Extract Dataset ZIP File

from google.colab import files
import zipfile
import os

# Upload the ZIP file
uploaded = files.upload()

# Unzip it
for file_name in uploaded.keys():
    if file_name.endswith(".zip"):
        with zipfile.ZipFile(file_name, 'r') as zip_ref:
            zip_ref.extractall("/content/Dataset")

# Confirm folders
os.listdir("/content/Dataset")

# STEP 2 - Import Required Libraries

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# STEP 3 - Define Directory Paths

base_dir = "/content/Dataset/Dataset"
train_path = os.path.join(base_dir, "Train")
valid_path = os.path.join(base_dir, "Valid")
test_path = os.path.join(base_dir, "Test")

# STEP 4 - Dataset Analysis

# Count images per class in training set
class_counts = {
    cls: len(os.listdir(os.path.join(train_path, cls)))
    for cls in os.listdir(train_path)
    if os.path.isdir(os.path.join(train_path, cls))
}
print("📊 Image count per class:", class_counts)

# Bar plot
plt.figure(figsize=(8, 5))
sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()))
plt.title("Image Count per Class (Training Set)")
plt.xlabel("Class")
plt.ylabel("Image Count")
plt.show()

# STEP 5 - Data Augmentation, Preprocessing & Building a Custom CNN Model

# Import Required Libraries
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import layers, models

# Set Paths to Your Dataset
train_path = "/content/Dataset/Dataset/Train"
valid_path = "/content/Dataset/Dataset/Valid"

# Define Data Augmentation and Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

valid_datagen = ImageDataGenerator(rescale=1./255)

# Create Data Generators
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

valid_generator = valid_datagen.flow_from_directory(
    valid_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Build Custom CNN Model with Explicit Name
custom_model = models.Sequential(name="Sequential")
custom_model.add(layers.Input(shape=(224, 224, 3)))
custom_model.add(layers.Conv2D(32, (3, 3), activation='relu'))
custom_model.add(layers.MaxPooling2D((2, 2)))
custom_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
custom_model.add(layers.MaxPooling2D((2, 2)))
custom_model.add(layers.Conv2D(128, (3, 3), activation='relu'))
custom_model.add(layers.MaxPooling2D((2, 2)))
custom_model.add(layers.Flatten())
custom_model.add(layers.Dense(128, activation='relu'))
custom_model.add(layers.Dropout(0.5))
custom_model.add(layers.Dense(train_generator.num_classes, activation='softmax'))

# Compile the Model
custom_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Print Model Summary
custom_model.summary()

# Define Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)

# STEP 6 - Train the Custom CNN

history = custom_model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=10,
    callbacks=[early_stop, checkpoint],
    steps_per_epoch=len(train_generator),
    validation_steps=len(valid_generator),
    verbose=1
)

# STEP 7 - Evaluate Custom Model

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Path to test dataset
test_path = "/content/Dataset/Dataset/Test"

# Define test data generator
test_datagen = ImageDataGenerator(rescale=1./255)

test_gen = test_datagen.flow_from_directory(
    test_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Evaluate model on test set
loss, acc = custom_model.evaluate(test_gen)
print(f"🧪 Test Accuracy (Custom CNN): {acc:.4f}")

# Get predictions
y_pred_probs = custom_model.predict(test_gen)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_gen.classes

# Class labels
class_labels = list(test_gen.class_indices.keys())

# Classification report with zero_division fix
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_labels, zero_division=1))

# Confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix - Custom CNN")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

# STEP 8 - Transfer Learning

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Sequential

# Load MobileNetV2 base model without top layers
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze the base model layers

# Build the transfer learning model
model_tl = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

# Compile the model
model_tl.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Display model architecture
model_tl.summary()

# STEP 9 - Train Transfer Learning Model

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set paths
train_path = "/content/Dataset/Dataset/Train"
val_path = "/content/Dataset/Dataset/Valid"

# Image preprocessing
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

# Train generator
train_gen = train_datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Validation generator
val_gen = val_datagen.flow_from_directory(
    val_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

history_tl = model_tl.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    callbacks=[
        EarlyStopping(patience=3, restore_best_weights=True),
        ModelCheckpoint("mobilenetv2_model.keras", save_best_only=True)
    ]
)

# STEP 10 - Compare Models

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Custom Train')
plt.plot(history.history['val_accuracy'], label='Custom Val')
plt.plot(history_tl.history['accuracy'], label='TL Train')
plt.plot(history_tl.history['val_accuracy'], label='TL Val')
plt.title("Accuracy Comparison")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Custom Train')
plt.plot(history.history['val_loss'], label='Custom Val')
plt.plot(history_tl.history['loss'], label='TL Train')
plt.plot(history_tl.history['val_loss'], label='TL Val')
plt.title("Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()

# STEP 12 - Streamlit Application Deployment

!pip install streamlit pyngrok --quiet

app_code = '''
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
'''

# Save it to app.py
with open("app.py", "w") as f:
    f.write(app_code)

from pyngrok import ngrok
import time
ngrok.set_auth_token("2z0Oqv0tD166fELGCHwV2gLZwq1_2G2zUQRSs6C27k9vdzxwq")

# Run Streamlit App in background
!streamlit run app.py &> /content/logs.txt &

# Wait for app to boot
time.sleep(5)

# Get public URL
public_url = ngrok.connect(8501)
print("🚀 Streamlit running at:", public_url)
