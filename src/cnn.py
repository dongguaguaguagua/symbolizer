import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
from tqdm import tqdm

# SimpleCNN 类保持不变
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_accuracy(model, loader, topk=1):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.topk(outputs.data, topk, dim=1)
            total += labels.size(0)
            for i in range(labels.size(0)):
                if labels[i] in predicted[i]:
                    correct += 1
    return correct / total

def train_one_file(model, criterion, optimizer, file_path, batch_size, device):
    # 加载单个 .pt 文件
    data = torch.load(file_path)
    images = data['data']
    labels = data['labels'].long()  # 转换为 torch.long
    # 创建 DataLoader
    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(loader, desc=f"Training {os.path.basename(file_path)}", unit="batch"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        # print(inputs.shape)
        # print(outputs.shape)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total
    return epoch_loss, epoch_accuracy

def validate_model(model, criterion, test_loader, device):
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
    return test_loss, test_accuracy

if __name__ == "__main__":
    print("Training model...")
    num_classes = 370
    model = SimpleCNN(num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练数据路径
    data_dir = "../data/augmented_data/"
    # train_files = [f for f in os.listdir(data_dir) if f.startswith("train_") and f.endswith("_set.pt")]
    train_files = [f for f in os.listdir(data_dir) if f.startswith("HASYv2") and f.endswith(".pt")]
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

    # 保存模型
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_directory = '../models'
    model_filename = f'model_{current_time}.pth'
    full_model_path = os.path.join(model_directory, model_filename)
    os.makedirs(model_directory, exist_ok=True)
    torch.save(model.state_dict(), full_model_path)
    print(f"Model saved as: {model_filename}")
