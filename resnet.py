import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from datetime import datetime
from torchvision import models
from cnn import unpickle, get_loader, SimpleCNN


# Define ResNet model (use pre-trained weights as a starting point)
class HASYv2ResNet(nn.Module):
    def __init__(self, num_classes):
        super(HASYv2ResNet, self).__init__()
        self.resnet = models(weights=None)
        # Modify first layer for grayscale
        self.resnet.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # Modify last layer for num_classes
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


# Training function
def train_model(epoch, model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(
        train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"
    ):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        # print(inputs.shape)
        outputs = model(inputs)
        # print(outputs.shape, labels.shape)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total
    print(
        f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}"
    )


# Testing function
def test_model(model, test_loader):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss /= total
    test_accuracy = correct / total
    print(f"Validation Loss: {test_loss:.4f}, Validation Accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    # Load and process the data
    print("loading data...")
    HASYv2 = unpickle("./data/HASYv2")
    data = np.array(HASYv2["data"])
    labels = np.array(HASYv2["labels"])

    print("processing data...")
    train_loader, test_loader = get_loader(data, labels)

    num_classes = len(np.unique(labels)) + 1
    print(f"Number of classes: {num_classes}")

    # Initialize the model
    model = HASYv2ResNet(num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    model = model.to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train and evaluate the model
    num_epochs = 10
    for epoch in range(num_epochs):
        train_model(
            epoch, model, train_loader, criterion, optimizer, num_epochs=num_epochs
        )
        test_model(model, test_loader)

    # Save the model
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_directory = "./models"
    model_filename = f"resnet_{current_time}.pth"
    full_model_path = os.path.join(model_directory, model_filename)
    os.makedirs(model_directory, exist_ok=True)
    torch.save(model.state_dict(), full_model_path)
    print(f"model saved as: {model_filename}")
