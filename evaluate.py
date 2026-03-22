# evaluate.py
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from dataset_loader import load_fer2013
from advance_model import EmotionCNN

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

def evaluate_model(model, dataloader, device):

    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in dataloader:

            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            predictions = torch.argmax(F.softmax(outputs, dim=1), dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Create results folder
    os.makedirs("results1", exist_ok=True)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=emotion_labels
    )
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)

    plt.title("Confusion Matrix")
    plt.tight_layout()

    # Save confusion matrix
    plt.savefig("results1/confusion_matrix.png")

    plt.close()

    # Save metrics
    with open("results1/metrics.txt", "w") as f:

        f.write("Model Evaluation Metrics\n")
        f.write("-------------------------\n")

        f.write(f"Accuracy : {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall   : {recall:.4f}\n")
        f.write(f"F1 Score : {f1:.4f}\n")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EmotionCNN().to(device)

    model.load_state_dict(
        torch.load("models/emotion_model.pth", map_location=device)
    )

    _, test_loader = load_fer2013(batch_size=32)

    results = evaluate_model(model, test_loader, device)

    print("\nModel Performance:")
    for metric, value in results.items():
        print(f"{metric.capitalize()}: {value:.4f}")

    print("\nSaved results in 'results1/' folder")
