#train.py
import torch
import torch.optim as optim
import torch.nn as nn
from dataset_loader import load_fer2013
from model import EmotionCNN

# Hyperparameters
num_epochs = 10
learning_rate = 0.001
batch_size = 32

# Load dataset
train_loader, test_loader = load_fer2013(batch_size=batch_size)

# Initialize model, loss function, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Save trained model
torch.save(model.state_dict(), "models/emotion_model.pth")
print("Model saved successfully!")
