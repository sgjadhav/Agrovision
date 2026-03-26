import torch
from torchvision import transforms
from PIL import Image
import os
from model import AgroVisionModel

def predict_image(image_path):
    # 1. Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Initialize model and load downloaded weights
    # We set pretrained to False because we are loading our own custom trained weights
    model = AgroVisionModel(num_classes=5, pretrained=False)
    weights_path = os.path.join("models", "agrovision_tested.pth")
    
    # Load weights safely
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval() # Set to evaluation mode

    # 3. Prepare the image (same transformations as training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 4. Open image and apply transforms
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    # 5. Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)

    # 6. Map prediction to class name
    class_names = ["Healthy", "Brown Rust", "Yellow Rust", "Loose Smut","Septoria"]
    result = class_names[predicted.item()]
    
    print(f"Prediction: {result}")
    return result

if __name__ == '__main__':
    # Test it with a dummy image path
    test_image = "1st.jpeg"
    if os.path.exists(test_image):
        predict_image(test_image)
    else:
        print("Please place a test_leaf.jpg in your root folder to test.")
