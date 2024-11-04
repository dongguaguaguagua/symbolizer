import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime

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

print("loading data...")
HASYv2 = unpickle("./data/HASYv2")
data = np.array(HASYv2['data'])
labels = np.array(HASYv2['labels'])

print("processing data...")
data = data[:, :, 0, :]
data = data.transpose(2, 0, 1)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
# 进一步调整标签形状
y_train = y_train.flatten()
y_test = y_test.flatten()
# 转换为PyTorch张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)
# 创建数据集和数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print("training model...")
model = SimpleCNN()
# 设置损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# 训练模型
num_epochs = 1  # 可以根据需要调整
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()  # 清除梯度
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        running_loss += loss.item()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')

print("evaluating model...")
model.eval()  # 切换到评估模式
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.topk(outputs.data, 5, dim=1)
        total += labels.size(0)
        for i in range(labels.size(0)):
            if labels[i] in predicted[i]:
                correct += 1

accuracy = correct / total
print(f'Accuracy on test set: {accuracy * 100:.2f}%')

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
model_filename = f'model_{current_time}.pth'
torch.save(model.state_dict(), model_filename)
print(f"model saved as: {model_filename}")
