# Dataset_structure.py
import os

root1 = 'data/fer2013_data/test'  
valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')

for emotion in os.listdir(root1):
    emotion_path = os.path.join(root1, emotion)
    if os.path.isdir(emotion_path):
        files = os.listdir(emotion_path)
        image_files = [f for f in files if f.lower().endswith(valid_exts)]
        print(f"test {emotion}: {len(image_files)} valid image files")
        if not image_files:
            print(f"⚠️  No valid image files in: {emotion_path}")

root2 = 'data/fer2013_data/train'  # Change to test/train if needed
valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')

for emotion in os.listdir(root2):
    emotion_path = os.path.join(root2, emotion)
    if os.path.isdir(emotion_path):
        files = os.listdir(emotion_path)
        image_files = [f for f in files if f.lower().endswith(valid_exts)]
        print(f" train {emotion}: {len(image_files)} valid image files")
        if not image_files:
            print(f"⚠️  No valid image files in: {emotion_path}")
            
