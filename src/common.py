import os
import torch
import pickle
import numpy as np
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset, random_split
import seaborn as sns
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_directory = "./models"


def unpickle(file):
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")

    return dict


def load_data(data_path):
    data = unpickle(data_path)
    # print(data.keys())
    _data = np.array(data["data"])
    _labels = np.array(data["labels"])
    print("data loaded.")
    return _data, _labels


def get_loader(data, labels, batch_size=64, train_ratio=0.8):
    data = np.transpose(data, (3, 2, 0, 1))  # 变为 (168233, 3, 32, 32)

    data = torch.tensor(data, dtype=torch.float32)
    labels = torch.tensor(labels.squeeze(), dtype=torch.long)

    dataset = TensorDataset(data, labels)

    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def save_model(model, path="model"):
    current_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    model_filename = f"{path}_{current_time}.pth"
    full_model_path = os.path.join(model_directory, model_filename)
    os.makedirs(model_directory, exist_ok=True)
    torch.save(model.state_dict(), full_model_path)
    print(f"model saved as: {model_filename}")


# Training function
def train_model(model, train_loader, criterion, optimizer, epoch=0, num_epochs=10):
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
    return epoch_loss, epoch_accuracy


# Testing function
def test_model(model, test_loader, criterion):
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
    return test_loss, test_accuracy


def plot_map(mAP):
    try:
        x = list(range(len(mAP)))
        plt.plot(x, mAP, label='mAp')
        plt.xlabel('epoch')
        plt.ylabel('mAP')
        plt.title('Eval mAP')
        plt.xlim(0, len(mAP))
        plt.legend(loc='best')
        plt.savefig('./mAP.png')
        plt.close()
        print("successful save mAP curve!")
    except Exception as e:
        print(e)


def vitualize_loss(train_loss, val_loss):
    # 示例数据（你可以替换成自己的数据）
    epochs = np.arange(1, len(train_loss) + 1)

    # 设置Seaborn风格
    sns.set(style="whitegrid")

    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=epochs, y=train_loss, label='Train Loss', marker='o')
    sns.lineplot(x=epochs, y=val_loss, label='Validation Loss', marker='s')

    # 图表标题和标签
    plt.title("Training and Validation Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # 显示图像
    plt.show()
