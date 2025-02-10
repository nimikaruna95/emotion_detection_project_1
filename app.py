import streamlit as st
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from model import EmotionCNN
from face_detection import detect_face

# Load model
device = torch.device("cpu")
model = EmotionCNN()
model.load_state_dict(torch.load("models/emotion_model.pth", map_location=device))
model.eval()

# Define transformation
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Emotion labels
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

st.title("Emotion Detection from Images")
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    face = detect_face(img)
    
    if face is not None:
        face = cv2.resize(face, (48, 48))
        face = transform(face).unsqueeze(0)
        
        with torch.no_grad():
            output = model(face)
            _, predicted = torch.max(output, 1)
        
        st.image(img, caption="Uploaded Image", use_column_width=True)
        st.write(f"Detected Emotion: {emotions[predicted.item()]}")
    else:
        st.write("No face detected.")
