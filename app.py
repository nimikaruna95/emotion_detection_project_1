import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from model import EmotionCNN
from evaluate import evaluate_model
from dataset_loader import load_fer2013

# Load trained model
@st.cache_resource
def load_model(model_path="models/emotion_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

# Preprocess uploaded image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Predict emotion from image
def predict_emotion(model, image, device):
    image = image.to(device)
    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()  # Ensure it returns an index
    return predicted_class

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Streamlit UI
def main():
    st.set_page_config(page_title="Emotion Detection App", layout="centered")
    st.title("😊 Emotion Detection App")
    st.write("Upload an image to detect the emotion.")

    model, device = load_model()

    # Sidebar for Model Performance Evaluation
    st.sidebar.header("📊 Model Performance Metrics")
    if st.sidebar.button("Evaluate Model"):
        st.sidebar.write("Evaluating... ⏳")
        _, test_loader = load_fer2013(batch_size=32)
        results = evaluate_model(model, test_loader, device)
        
        st.sidebar.success("Evaluation Completed ✅")
        for metric, value in results.items():
            st.sidebar.write(f"**{metric.capitalize()}**: {value:.4f}")

    # Image Upload Section
    uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess and Predict
        image_tensor = preprocess_image(image).to(device)
        predicted_class = predict_emotion(model, image_tensor, device)

        # Display Prediction
        predicted_emotion = emotion_labels[predicted_class]
        st.markdown(f"### 🎭 Predicted Emotion: **{predicted_emotion}**")

if __name__ == "__main__":
    main()
