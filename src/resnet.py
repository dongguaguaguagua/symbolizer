import os
import torch
from torch.utils.data import DataLoader, TensorDataset
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
from cnn import train_one_file, validate_model

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
    # print("loading data...")
    # data, labels = load_data("./data/HASYv2")

    # print("processing data...")
    # train_loader, test_loader = get_loader(data, labels)

    num_classes = 370
    print(f"Number of classes: {num_classes}")

    # Initialize the model
    print("initializing model...")
    model = HASYv2ResNet(num_classes)

    print(f"Training on {device}")
    model = model.to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # # Train and evaluate the model
    # num_epochs = 10
    # for epoch in range(num_epochs):
    #     print(f"\nEpoch {epoch+1}/{num_epochs}")
    #     train_model(
    #         model, train_loader, criterion, optimizer
    #     )
    #     test_model(model, test_loader, criterion)

    # 训练数据路径
    data_dir = "../data/augmented_data/"
    train_files = [f for f in os.listdir(data_dir) if f.startswith("train_") and f.endswith("_set.pt")]
    test_file = os.path.join(data_dir, "test_set.pt")

    # 加载测试集
    test_data = torch.load(test_file)
    test_images = test_data['data']
    test_labels = test_data['labels'].long()  # 转换为 torch.long
    test_dataset = TensorDataset(test_images, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 训练参数
    num_epochs = 1  # 每个文件训练的epoch数
    batch_size = 64

    # 开始训练
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # 逐个文件训练
        for train_file in train_files:
            file_path = os.path.join(data_dir, train_file)
            loss, accuracy = train_one_file(model, criterion, optimizer, file_path, batch_size, device)
            print(f"File: {train_file}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

        # 在测试集上验证
        test_loss, test_accuracy = validate_model(model, criterion, test_loader, device)
        print(f"Validation Loss: {test_loss:.4f}, Validation Accuracy: {test_accuracy:.4f}")


    # Save the model
    save_model(model, "resnet")
