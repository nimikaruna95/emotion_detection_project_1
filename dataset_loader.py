#dataset_loader.py
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

def load_fer2013(batch_size=32):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale (FER-2013 is grayscale)
        transforms.Resize((48, 48)),  # Resize images to 48x48
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize pixel values
    ])

    # Load dataset using ImageFolder
    train_set = ImageFolder(root="data/fer2013_data/train", transform=transform)
    test_set = ImageFolder(root="data/fer2013_data/test", transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# Run this file to check if dataset loads correctly
if __name__ == "__main__":
    train_loader, test_loader = load_fer2013()
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Test dataset size: {len(test_loader.dataset)}")

