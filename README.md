# 🧠 BRAIN TUMOR MRI IMAGE CLASSIFICATION

This project implements a **complete end-to-end pipeline** for classifying brain tumor MRI images using both a **Custom Convolutional Neural Network (CNN)** and a **Transfer Learning** model (MobileNetV2). It includes data preprocessing, augmentation, training, evaluation, model comparison, and a Streamlit web app for live prediction.

---

## 🔧 Steps

| Step | Description |
|------|-------------|
|1️⃣ | Upload & Extract MRI Dataset (ZIP) |
| 2️⃣ | Preprocess and Augment Images using `ImageDataGenerator` |
| 3️⃣ | Build a Custom CNN architecture |
| 4️⃣ | Train CNN with EarlyStopping and ModelCheckpoint |
| 5️⃣ | Evaluate model on test set and visualize confusion matrix |
| 6️⃣ | Build Transfer Learning model using **MobileNetV2** |
| 7️⃣ | Compare both models using accuracy/loss curves |
| 8️⃣ | Deploy model via a **Streamlit Web App** |

---

## 🏗️ Model Architectures

### 🔹 Custom CNN
- 3 Convolutional + MaxPooling layers
- Fully connected Dense + Dropout
- Softmax output for 4 classes

### 🔹 Transfer Learning (MobileNetV2)
- Pre-trained on ImageNet
- `include_top=False` for feature extraction
- Additional GAP + Dense + Dropout + Output layers

---

## 📊 Evaluation

Both models are evaluated on accuracy, loss, classification report, and confusion matrix. Plots are generated to compare model performance across epochs.

---

## 🛠️ Requirements

- Python ≥ 3.7
- TensorFlow ≥ 2.x
- Keras
- Streamlit
- Scikit-learn
- Matplotlib
- Seaborn
- NumPy
- Plotly (for visualization in dashboards)

---

## 🧪 Streamlit App

An interactive app is built using **Streamlit**:
- Upload a brain MRI image
- Get instant predictions from -
  - ✅ Custom CNN
  - ✅ MobileNetV2
- Display prediction probabilities and result


<p align="center">
  <img src="Streamlit Images/1.png" alt="Streamlit Screenshot 1" width="80%">
</p>

<table>
  <tr>
    <td><img src="Streamlit Images/2.png" width="100%"></td>
    <td><img src="Streamlit Images/3.png" width="100%"></td>
  </tr>
  <tr>
    <td><img src="Streamlit Images/4.png" width="100%"></td>
    <td><img src="Streamlit Images/5.png" width="100%"></td>
  </tr>
  <tr>
    <td><img src="Streamlit Images/6.png" width="100%"></td>
    <td><img src="Streamlit Images/7.png" width="100%"></td>
  </tr>
  <tr>
    <td><img src="Streamlit Images/8.png" width="100%"></td>
    <td><img src="Streamlit Images/9.png" width="100%"></td>
  </tr>
</table>

