import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dataset_loader import load_fer2013
from model import EmotionCNN

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
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionCNN().to(device)
    model.load_state_dict(torch.load("models/emotion_model.pth", map_location=device))
    
    _, test_loader = load_fer2013(batch_size=32)
    results = evaluate_model(model, test_loader, device)
    
    print("Model Performance:")
    for metric, value in results.items():
        print(f"{metric.capitalize()}: {value:.4f}")
