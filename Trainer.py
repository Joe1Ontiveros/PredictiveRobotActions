import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from PIL import Image

def auto_label_images(image_dir):
    data = []
    for root, dirs, files in os.walk(image_dir):
        label = None
        if 'label.txt' in files:
            with open(os.path.join(root, 'label.txt')) as f:
                label = f.read().strip()
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                if label is None:
                    label = os.path.basename(root)
                data.append((img_path, label))
    return data


class YOLOv8_PredictURobot:
    def __init__(self):
        from ultralytics import YOLO
        # Load a model
        self.model = YOLO("yolov8n.pt")

    def initiate_training(self,yaml_data_input:str):
        # Train the model on our set
        self.train_results = self.model.train(
            data=yaml_data_input,  # path to dataset YAML
            epochs=100,  # number of training epochs
            imgsz=640,  # training image size
            device="cpu",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
        )
    
    def evaluate_md():
        metrics = model.val()

    def action_model():
        # Perform object detection on an image
        results = model("path/to/image.jpg")
        results[0].show()

        # Export the model to ONNX format
        path = model.export(format="onnx")  # return path to exported model

class ImageLabelDataset(Dataset):
    def __init__(self, data, label_to_idx, transform=None):
        self.data = data
        self.label_to_idx = label_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label_idx = self.label_to_idx[label]
        return image, label_idx

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def get_model(num_classes):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def train_model(model, dataloader, num_epochs=10, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        model.train()
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
    return model

# Example usage:
data = auto_label_images('path/to/dataset')
labels = sorted(set(label for _, label in data))
label_to_idx = {label: idx for idx, label in enumerate(labels)}
dataset = ImageLabelDataset(data, label_to_idx, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
model = get_model(num_classes=len(labels))
model = train_model(model, dataloader, num_epochs=10)
