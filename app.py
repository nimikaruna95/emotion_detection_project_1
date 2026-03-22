# app.py
import streamlit as st
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import torchvision.transforms as transforms
import time

from update_model import EmotionCNN

# Emotion labels
emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

# Mediapipe setup
mp_face = mp.solutions.face_detection
mp_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

# Haarcascade
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

MAX_FILE_SIZE = 5 * 1024 * 1024  # image upto 5MB

# Model
@st.cache_resource
def load_model():
    model = EmotionCNN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load("models1/emotion_model.pth", map_location=device))
    model.eval()
    return model, device

# Face Detection
def detect_faces_mediapipe(img):
    faces = []
    with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6) as fd:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = fd.process(rgb)

        if results.detections:
            for det in results.detections:
                bbox = det.location_data.relative_bounding_box
                h, w, _ = img.shape
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                w_box = int(bbox.width * w)
                h_box = int(bbox.height * h)
                faces.append((x, y, w_box, h_box))
    return faces


def detect_faces_haar(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 🔥 Resize for better detection
    gray = cv2.resize(gray, (300, 300))

    faces = haar_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,     # more precise
        minNeighbors=3,      # reduce strictness
        minSize=(30, 30)     # detect smaller faces
    )

    # Convert back to original scale
    h_ratio = img.shape[0] / 300
    w_ratio = img.shape[1] / 300

    scaled_faces = []
    for (x, y, w, h) in faces:
        scaled_faces.append((
            int(x * w_ratio),
            int(y * h_ratio),
            int(w * w_ratio),
            int(h * h_ratio)
        ))

    return scaled_faces

# Landmarks
def draw_landmarks(img):
    with mp_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as mesh:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = mesh.process(rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_draw.draw_landmarks(
                    img,
                    face_landmarks,
                    mp_mesh.FACEMESH_TESSELATION
                )
    return img

def extract_landmark_mask(img):
    mask = np.zeros_like(img)

    with mp_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as mesh:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = mesh.process(rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_draw.draw_landmarks(
                    mask,
                    face_landmarks,
                    mp_mesh.FACEMESH_TESSELATION
                )
    return mask

# Preprocessing
def crop_face(img, bbox):
    x, y, w, h = bbox
    margin = 20

    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(img.shape[1], x + w + margin)
    y2 = min(img.shape[0], y + h + margin)

    face = img[y1:y2, x1:x2]
    face = cv2.resize(face, (48, 48))

    return Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

def preprocess(img):
    transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    return transform(img).unsqueeze(0)

# Prediction
def predict(model, img, device):
    with torch.no_grad():
        out = model(img.to(device))
        pred = torch.argmax(F.softmax(out, 1), 1).item()
    return pred

# Main Function
def main():
    st.title("🔥 Advanced Emotion Detection System")
    st.write("Upload an image containing a face")

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is None:
        return

    if uploaded_file.size > MAX_FILE_SIZE:
        st.error("File size exceeds 5MB")
        return

    try:
        img_pil = Image.open(uploaded_file).convert("RGB")
    except:
        st.error("Invalid image file")
        return

    img_np = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    st.image(img_pil, caption="Uploaded Image")

    # Load model
    model, device = load_model()

    # Choosing the detection method
    method = st.radio("Choose Face Detection Method", ["MediaPipe", "Haarcascade"])

    # Face Detection 
    start_time = time.time()

    if method == "MediaPipe":
        faces = detect_faces_mediapipe(img_np)
    else:
        faces = detect_faces_haar(img_np)

    detection_time = time.time() - start_time

    st.write(f"⏱ Detection Time: {detection_time:.4f} sec")

    if len(faces) == 0:
        st.warning("No face detected")
        return

    st.success(f"{len(faces)} face(s) detected using {method}")

    # Draw landmarks
    img_landmarks = draw_landmarks(img_np.copy())
    st.image(cv2.cvtColor(img_landmarks, cv2.COLOR_BGR2RGB), caption="Facial Landmarks")

    # Face Looping process
    for i, bbox in enumerate(faces):

        face = crop_face(img_np, bbox)
        st.image(face, caption=f"Face {i+1}")

        # Normal Prediction
        start_pred = time.time()

        img_tensor = preprocess(face)
        pred_normal = predict(model, img_tensor, device)

        pred_time = time.time() - start_pred

        # Landmark prediction
        face_np = cv2.cvtColor(np.array(face), cv2.COLOR_RGB2BGR)
        landmark_img = extract_landmark_mask(face_np)

        landmark_pil = Image.fromarray(cv2.cvtColor(landmark_img, cv2.COLOR_BGR2RGB))
        st.image(landmark_pil, caption="Landmark Image")

        img_tensor_landmark = preprocess(landmark_pil)
        pred_landmark = predict(model, img_tensor_landmark, device)

        # Results
        st.success(f"Normal Prediction: {emotion_labels[pred_normal]}")
        st.info(f"Landmark Prediction: {emotion_labels[pred_landmark]}")

        # Analysis 
        if pred_normal == pred_landmark:
            st.write("✔ Landmark did NOT change prediction")
        else:
            st.warning("⚠ Landmark changed prediction → impacts model")

        st.write(f"⏱ Prediction Time: {pred_time:.4f} sec")

        # FPS
        total_time = detection_time + pred_time
        fps = 1 / total_time if total_time > 0 else 0
        st.write(f"⚡ FPS: {fps:.2f}")


if __name__ == "__main__":
    main()
