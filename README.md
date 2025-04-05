
# Emotion Detection System using CNN and Streamlit

This project is a deep learning-based Emotion Detection System that uses Convolutional Neural Networks (CNNs) trained on the FER-2013 dataset and is deployed with a user-friendly interface using Streamlit.

---

## Project Structure

```
Emotion_detection/
│
├── data/
│   ├── fer2013.zip
│   ├── fer2013_data/
│   │   ├── train/
│   │   ├── train_detected/
│   │   ├── test/
│   │   └── test_detected/
│
├── models/
│   └── emotion_model.pth
│
├── app.py                  ← Streamlit UI
├── train.py                ← Model training
├── model.py                ← CNN architecture
├── face_detection.py       ← Dlib-based face detector
├── dataset_loader.py       ← Loads and preprocesses data
└── evaluate.py             ← Model evaluation script ✅
```

---

## Concepts Used

| Concept                 | Description                                           | Where it Appears |
|--------                 |-------------                                          |------------------|
| **CNN**                 | For extracting visual features to classify emotion.   | `model.py`, `train.py`, `app.py`, `evaluate.py` |
| **Streamlit**           | Python framework for building web UIs easily.         | `app.py` |
| **Face Detection**      | Detect faces in images before classification.         | `face_detection.py` |
| **FER-2013 Dataset**    | Benchmark dataset with 7 emotions.                    | `dataset_loader.py`, `train.py`, `evaluate.py` |
| **Image Preprocessing** | Resize, grayscale, normalization.                     | `dataset_loader.py`, `app.py` |
| **Model Evaluation**    | Accuracy, precision, recall, F1 score.                | `evaluate.py`, `app.py` |
| **PyTorch**             | Deep learning framework.                              | All model-related files |


---

## Files Explained

### `app.py`
- Streamlit web interface
- Upload image and predict emotion
- Calls model, preprocessing, evaluation functions

### `train.py`
- Defines training loop
- Trains CNN using FER-2013 dataset
- Saves model to `models/emotion_model.pth`

### `model.py`
- Contains the CNN architecture
- 3 Conv layers + MaxPool + FC layers
- Output: 7 emotion classes

### `face_detection.py`
- Uses Dlib to detect and crop faces
- Saves new dataset with cropped face regions

### `evaluate.py`
- Evaluates model using precision, recall, F1, accuracy
- Prints classification report

### `dataset_loader.py`
- Prepares dataset using PyTorch `ImageFolder`
- Applies preprocessing (resize, grayscale, normalize)

---

## How to Run

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Train the model
python train.py

# Step 3: Run the web app
streamlit run app.py
```

---

## 📊 Supported Emotions
- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

---
