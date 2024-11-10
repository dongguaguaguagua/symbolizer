import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import models
from common import (
    load_data,
    get_loader,
    save_model,
    train_model,
    test_model,
    device,
)


# Define ResNet model (use pre-trained weights as a starting point)
class HASYv2ResNet(nn.Module):
    def __init__(self, num_classes):
        super(HASYv2ResNet, self).__init__()
        self.resnet = models.resnet34(weights=None)
        # Modify first layer for grayscale
        self.resnet.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # Modify last layer for num_classes
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


if __name__ == "__main__":
    print("loading data...")
    data, labels = load_data("./data/HASYv2")

    print("processing data...")
    train_loader, test_loader = get_loader(data, labels)

    num_classes = len(np.unique(labels)) + 1
    print(f"Number of classes: {num_classes}")

    # Initialize the model
    print("initializing model...")
    model = HASYv2ResNet(num_classes)
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
        test_model(model, test_loader, criterion)

    # Save the model
    save_model(model, "resnet")
