#face_detection.py
import cv2
import dlib
import os

# Function to detect faces in the image and save the output
def detect_face_and_save(image_path, output_dir):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return

    # Convert to grayscale (dlib works with grayscale images)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initialize face detector (dlib)
    detector = dlib.get_frontal_face_detector()

    # Detect faces
    faces = detector(gray)
    print(f"Number of faces detected in {image_path}: {len(faces)}")

    # Draw rectangles around faces in the original image
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Save the output image with the detected faces
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, image)
    print(f"Saved output image: {output_path}")

# Directories containing images for train and test data
base_dir_train = 'data/fer2013_data/train'  # Replace with your train data folder path
base_dir_test = 'data/fer2013_data/test'   # Replace with your test data folder path

# Output directories for detected faces
output_base_dir_train = 'data/fer2013_data/train_detected'  # Output folder for train images with faces detected
output_base_dir_test = 'data/fer2013_data/test_detected'    # Output folder for test images with faces detected

# Create output directories if they don't exist
if not os.path.exists(output_base_dir_train):
    os.makedirs(output_base_dir_train)

if not os.path.exists(output_base_dir_test):
    os.makedirs(output_base_dir_test)

# List of emotion folders (same for both train and test datasets)
emotion_folders = ['angry', 'disgust', 'fear', 'sad', 'neutral', 'happy', 'surprise']

# Function to process dataset (train or test)
def process_dataset(base_dir, output_base_dir):
    for emotion in emotion_folders:
        emotion_folder_path = os.path.join(base_dir, emotion)
        output_emotion_folder = os.path.join(output_base_dir, emotion)
        
        # Create an output folder for each emotion category
        if not os.path.exists(output_emotion_folder):
            os.makedirs(output_emotion_folder)

        # Loop through all images in the folder
        for filename in os.listdir(emotion_folder_path):
            if filename.endswith(('.jpg', '.png')):  # Only process images with specific extensions
                image_path = os.path.join(emotion_folder_path, filename)
                detect_face_and_save(image_path, output_emotion_folder)

# Process bot)h the train and test datasets
print("Processing Train Dataset...")
process_dataset(base_dir_train, output_base_dir_train)

print("Processing Test Dataset...")
process_dataset(base_dir_test, output_base_dir_test)

print("Face detection processing complete!")

