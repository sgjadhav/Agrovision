import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import os
from model import AgroVisionModel

# 1. Page Configuration
st.set_page_config(page_title="AgroVision AI", page_icon="🌿", layout="centered")

# 2. Cache the model so it doesn't reload and freeze the app on every click
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize the model with 5 classes to match your Kaggle training
    model = AgroVisionModel(num_classes=5, pretrained=False)
    weights_path = os.path.join("models", "agrovision_tested.pth")
    
    # Load the weights safely
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

# 3. Image Preprocessing Function
def process_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# 4. Main App UI
def main():
    st.title("🌿 AgroVision: Crop Disease Detector")
    st.write("Upload an image of a wheat leaf to detect its health status.")

    # Load the model in the background
    model, device = load_model()
    
    # Ensure these are alphabetical to match Kaggle's folder structure
    class_names = ["Brown Rust", "Healthy", "Loose Smut", "Septoria", "Yellow Rust"]

    # Create a file uploader widget
    uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Leaf", use_container_width=True)

        # Create a big prediction button
        if st.button("Predict Disease", type="primary"):
            with st.spinner("Analyzing leaf with EfficientNet & CBAM..."):
                
                # Prepare image and predict
                img_tensor = process_image(image).to(device)
                
                with torch.no_grad():
                    outputs = model(img_tensor)
                    # Convert raw scores to percentages (probabilities)
                    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                    confidence, predicted_idx = torch.max(probabilities, 0)
                
                # Extract the final result and confidence score
                result = class_names[predicted_idx.item()]
                conf_score = confidence.item() * 100

                # Display Results beautifully on the screen
                st.success(f"**Prediction:** {result}")
                st.info(f"**Confidence:** {conf_score:.2f}%")

if __name__ == '__main__':
    main()