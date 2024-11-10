import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
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


class EfficientNetB0Model(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetB0Model, self).__init__()
        # 加载预训练的EfficientNet-B0模型
        self.model = models.efficientnet_b0(weights="EfficientNet_B0_Weights.DEFAULT")
        # 替换最后一层以适应我们的分类任务
        self.model.classifier[1] = nn.Linear(
            self.model.classifier[1].in_features, num_classes
        )

    def forward(self, x):
        return self.model(x)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=370):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 输入1个通道
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # 32x32 -> 16x16 -> 8x8
        self.fc2 = nn.Linear(128, num_classes)  # 共有369个字符

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x.unsqueeze(1))))  # 添加通道维度
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)  # 展平
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class StudentCNN(nn.Module):
    def __init__(self, num_classes):
        super(StudentCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 输入3个通道
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # 32x32 -> 16x16 -> 8x8
        self.fc2 = nn.Linear(128, num_classes)  # 共有369个字符

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 添加通道维度
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)  # 展平
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_accuracy(model, loader, topk):
    model.eval()  # 切换到评估模式
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            _, predicted = torch.topk(outputs.data, topk, dim=1)
            total += labels.size(0)
            for i in range(labels.size(0)):
                if labels[i] in predicted[i]:
                    correct += 1
    accuracy = correct / total
    return accuracy


if __name__ == "__main__":
    print("loading data...")
    data, labels = load_data("./data/HASYv2")

    print("processing data...")
    train_loader, test_loader = get_loader(data, labels)
    num_classes = len(np.unique(labels)) + 1

    # data = data[:, :, 0, :]
    # data = data.transpose(2, 0, 1)
    # X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    # # 进一步调整标签形状
    # y_train = y_train.flatten()
    # y_test = y_test.flatten()
    # # 转换为PyTorch张量
    # X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    # y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    # X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    # y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    # # 创建数据集和数据加载器
    # train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    # test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print("initializing model...")
    # model = SimpleCNN(num_classes)
    model = EfficientNetB0Model(num_classes)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train and evaluate the model
    print("training model...")
    num_epochs = 10
    for epoch in range(num_epochs):
        train_model(
            epoch, model, train_loader, criterion, optimizer, num_epochs=num_epochs
        )
        test_model(model, test_loader, criterion)
    print("saving model...")
    save_model(model, "EfficientNetB0")
