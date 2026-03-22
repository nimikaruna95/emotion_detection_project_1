
# Emotion Detection System using CNN and Streamlit

This project is a deep learning-based Emotion Detection System that uses Convolutional Neural Networks (CNNs) trained on the FER-2013 dataset and is deployed with a user-friendly interface using Streamlit.

---

## Project Structure

```
emotion-detection/
│
├── app.py                      # Streamlit web application
├── model.py                    # CNN model architecture
├── train.py                    # Model training
├── evaluate.py                 # Model evaluation & metrics
├── dataset_loader.py           # Data loading & preprocessing
├── data_structure.py           # Dataset validation
├── requirements.txt            # Required libraries and dependencies
│
├── models/                    # Saved trained models
│   └── emotion_model.pth
│
├── results/                   # Training results (graphs)
│   ├── accuracy_graph.png
│   └── loss_graph.png
│
├── results1/                   # Evaluation results
│   ├── confusion_matrix.png
│   └── metrics.txt
│
└── data/
    └── fer2013_data/
        ├── train/              # Training dataset
        │   ├── Angry/
        │   ├── Disgust/
        │   ├── Fear/
        │   ├── Happy/
        │   ├── Neutral/
        │   ├── Sad/
        │   └── Surprise/
        │
        └── test/               # Testing dataset
            ├── Angry/
            ├── Disgust/
            ├── Fear/
            ├── Happy/
            ├── Neutral/
            ├── Sad/
            └── Surprise/

---

## Concepts Used

----------------------------------------------------------------------------------------------------------------
| Concept               | Description                                         | Where it Appears               |
|-----------------------|-----------------------------------------------------|--------------------------------|
| CNN                   | For extracting visual features to classify emotion. | model.py                       |
| Streamlit             | Python framework for building web UIs easily.       | app.py                         |
| Face Detection        | Detect faces in images before classification.       | app.py                         |
| FER-2013 Dataset      | Benchmark dataset with 7 emotions.                  | dataset_loader.py              |
| Image Preprocessing   | Resize, grayscale, normalization.                   | dataset_loader.py, app.py      |
| Model Evaluation      | Accuracy, precision, recall, F1-score.              | evaluate.py                    |
| PyTorch               | Deep learning framework used for modeling.          | All model-related files        |
| Landmark Detection    | Extract facial keypoints using MediaPipe.           | app.py                         |
| Data Augmentation     | Improves generalization using flips & rotation.     | dataset_loader.py              |
| Optimizer (Adam)      | Updates model weights efficiently.                  | train.py                       |
| Loss Function         | Measures prediction error (CrossEntropy).           | train.py                       |
| Confusion Matrix      | Visualizes prediction performance.                  | evaluate.py                    |
| Performance Metrics   | Detection time, prediction time, FPS.               | app.py                         |
----------------------------------------------------------------------------------------------------------------
---

## Files Explained

### `app.py'
- Builds the Streamlit application that allows users to upload images, detects faces using MediaPipe or Haarcascade.
- Extracts landmarks, preprocesses images, and predicts emotions using the trained CNN model.

### `train.py`
- Evaluates the trained model using test data
- calculates metrics like accuracy, precision, recall, and F1-score, generates a confusion matrix
- saves evaluation results for performance analysis.
- Saves model to `models/emotion_model.pth`

### `model.py`
- Defines the EmotionCNN architecture using multiple convolutional and batch normalization layers.
- Applies pooling layers to reduce spatial dimensions and improve feature extraction.
- Uses fully connected layers to classify facial images into different emotion categories.

### `dataset_structure.py`
- Checks the dataset directory structure and validates image files in each emotion category.
- Ensuring proper dataset organization and identifying missing or invalid files before training the model.

### `evaluate.py`
- Evaluates the trained model using test data.
- calculates metrics like accuracy, precision, recall, and F1-score, generates a confusion matrix
- saves evaluation results for performance analysis.

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

## Emotions
- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

---
