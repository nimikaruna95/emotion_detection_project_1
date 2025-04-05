# Emotion Detection using CNN and Streamlit
A deep learning project for facial emotion recognition using Convolutional Neural Networks (CNNs) trained on the FER-2013 dataset. The system detects emotions from facial images and provides a Streamlit web interface for user interaction.
----------------------------------------------------------------------------------------------------------------------
## Project Structure
Emotion_detection/ 
│ ├── data/ 
│ ├── fer2013.zip # Original FER-2013 dataset 
│ ├── fer2013_data/ 
│ │ ├── train/ # Raw training images 
│ │ ├── train_detected/ # Face-detected training images 
│ │ ├── test/ # Raw test images 
│ │ └── test_detected/ # Face-detected test images 
│ ├── models/ 
│ └── emotion_model.pth # Trained CNN model 
│── app.py # Streamlit app interface 
├── train.py # CNN model training script 
├── model.py # CNN architecture 
├── face_detection.py # Dlib-based face detection preprocessing 
├── dataset_loader.py # Data loading utilities 
└── evaluate.py # Model evaluation (accuracy, precision, recall, F1)
-----------------------------------------------------------------------------------------------------------------------------------------
## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/emotion-detection
cd emotion-detection
-----------------------------------------------------------------------------------------------------------------------------------------
### 2. Install Dependencies
pip install -r requirements.txt
-------------------------------------
Main libraries:
-torch, torchvision
-opencv-python
-dlib
-numpy, PIL, sklearn
-streamlit
Note: For dlib, you may need CMake installed on your system.
--------------------------------------------------------------
Dataset
Uses FER-2013, a labeled dataset of 48x48 grayscale facial images across 7 emotion categories:
['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
--------------------------------------------------------------------
Structure:
Organize dataset folders like this:
fer2013_data/
├── train/
│   ├── Angry/
│   ├── Happy/
│   ...
├── test/
│   ├── Angry/
│   ├── Happy/
│   ...
----------------------------------------------------------------------------------------------------------------------------------------
Preprocessing
Run the face detection step to crop and align faces:
python face_detection.py
This uses dlib to detect and save aligned facial regions in:
-train_detected/
-test_detected/

