# train.py
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from dataset_loader import load_fer2013
from advance_model import EmotionCNN
from sklearn.utils.class_weight import compute_class_weight
import os

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Hyperparameters
num_epochs = 50
learning_rate = 0.001
batch_size = 32

# Loading the dataset
train_loader, test_loader = load_fer2013(batch_size=batch_size)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
model = EmotionCNN().to(device)

# Class weights
classes = np.array([0,1,2,3,4,5,6])
targets = [label for _, label in train_loader.dataset.samples]

class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=targets)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# To store metrices values
train_losses = []
train_accuracies = []

# Training
for epoch in range(num_epochs):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Calculation of Accuracy
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total

    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f} Accuracy: {epoch_acc:.4f}")

# Saving the model
torch.save(model.state_dict(), "models/emotion_model.pth")
print("Model saved successfully")

# Graphs
plt.figure()

plt.plot(train_accuracies, label="Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Accuracy Over Epochs")
plt.legend()

plt.savefig("results/accuracy_graph.png")
plt.close()

# Loss graph
plt.figure()

plt.plot(train_losses, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.legend()

plt.savefig("results/loss_graph.png")
plt.close()

print("Graphs saved in 'results/' folder")

