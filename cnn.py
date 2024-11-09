import os
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, TensorDataset, random_split
from datetime import datetime
from tqdm import tqdm

class EfficientNetB0Model(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetB0Model, self).__init__()
        # 加载预训练的EfficientNet-B0模型
        self.model = models.efficientnet_b0(weights = "EfficientNet_B0_Weights.DEFAULT")
        # 替换最后一层以适应我们的分类任务
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.model(x)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 输入1个通道
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # 32x32 -> 16x16 -> 8x8
        self.fc2 = nn.Linear(128, 370)  # 共有369个字符

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x.unsqueeze(1))))  # 添加通道维度
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)  # 展平
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


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


# 定义训练函数
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_accuracy = correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

        # 验证模型性能
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

def get_loader(data, labels):
    data = np.transpose(data, (3, 2, 0, 1)) # 变为 (168233, 3, 32, 32)

    data = torch.tensor(data, dtype=torch.float32)
    labels = torch.tensor(labels.squeeze(), dtype=torch.long)

    dataset = TensorDataset(data, labels)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

if __name__ == '__main__':
    print("loading data...")
    HASYv2 = unpickle("./data/HASYv2")
    data = np.array(HASYv2['data']) # (32, 32, 3, 168233)
    labels = np.array(HASYv2['labels']) # (168233, 1)
    print("processing data...")
    train_loader, test_loader = get_loader(data, labels)
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

    print("training model...")
    # model = SimpleCNN()
    model = EfficientNetB0Model(370)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=2)

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_directory = './models'
    model_filename = f'model_{current_time}.pth'
    full_model_path = os.path.join(model_directory, model_filename)
    os.makedirs(model_directory, exist_ok=True)
    torch.save(model.state_dict(), full_model_path)
    print(f"model saved as: {model_filename}")
