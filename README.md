# 🌾 AgroVision: AI-Powered Crop Pathology

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://agrovision-sj.streamlit.app/)

AgroVision is an end-to-end Deep Learning pipeline designed to diagnose wheat leaf diseases with high accuracy. Built with PyTorch and deployed via Streamlit, this tool allows farmers and agricultural researchers to instantly identify crop pathology from a simple image, enabling faster intervention and better yield protection.

## ✨ Live Demo
Try the live application here: **[AgroVision Web App](https://agrovision-sj.streamlit.app/)**

## 🧠 Technical Architecture
Unlike standard out-of-the-box models, AgroVision utilizes a custom-engineered architecture to maximize feature extraction from complex agricultural backgrounds:
* **Base Model:** `EfficientNet-B0` (Pre-trained)
* **Custom Attention Mechanism:** Integrated a **CBAM (Convolutional Block Attention Module)** layer directly after the feature extractor. This allows the model to actively focus on the diseased lesions on the leaf rather than memorizing the background soil or sky.
* **Classifier:** Custom fully connected layers optimized for 5 distinct wheat leaf classes.
* **Frameworks:** PyTorch, Torchvision, Streamlit.

## 📊 Dataset & Training Methodology
To ensure the model is robust and strictly evaluated without bias, the pipeline utilizes a rigorous training strategy:
* **Environment:** Trained on cloud GPUs.
* **Data Augmentation:** Implemented Random Horizontal Flips, Rotations, and Normalization on the training set to prevent overfitting.
* **Strict Split (70/15/15):** * **70% Training Set:** For weight optimization.
  * **15% Validation Set:** For hyperparameter tuning and early stopping.
  * **15% Blind Test Set:** A completely unseen dataset evaluated only *after* training was finalized to verify true real-world accuracy.

## 🚀 How to Run Locally

If you want to run this application on your own machine:

1. **Clone the repository**
   ```bash
   git clone [https://github.com/sgjadhav/agrovision-ai.git](https://github.com/sgjadhav/agrovision-ai.git)
   cd agrovision-ai
python -m venv venv
venv\Scripts\activate  # On Windows

source venv/bin/activate  # On Mac/Linux

pip install -r requirements.txt

streamlit run app.py
