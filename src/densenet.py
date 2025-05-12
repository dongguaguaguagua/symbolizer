from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from common import (
    load_data,
    get_loader,
    save_model,
    train_model,
    test_model,
    device,
)


# Define DenseNet-121 model (using no pre-trained weights)
class HASYv2DenseNet(nn.Module):
    def __init__(self, num_classes):
        super(HASYv2DenseNet, self).__init__()
        # Load DenseNet-121 architecture
        self.densenet = models.densenet121(weights=None)
        # Modify classifier layer for our number of classes
        in_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.densenet(x)


if __name__ == "__main__":
    # Hyperparameters
    num_classes = 370
    num_epochs = 10
    batch_size = 64
    learning_rate = 0.001

    # Load and prepare data
    print("Loading data...")
    data, labels = load_data("./data/HASYv2")
    print("Creating data loaders...")
    train_loader, test_loader = get_loader(data, labels, batch_size=batch_size)

    # Initialize model
    print(f"Initializing DenseNet-121 with {num_classes} classes...")
    model = HASYv2DenseNet(num_classes).to(device)
    print(f"Model will train on device: {device}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_model(model, train_loader, criterion, optimizer)
        test_model(model, test_loader, criterion)

    save_model(model, "densenet121")
    print("Training complete. Model saved.")
