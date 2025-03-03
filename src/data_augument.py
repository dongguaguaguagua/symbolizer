import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F
import numpy as np
from PIL import Image
import os
from cnn import unpickle
import json
from frontend import *
import pandas as pd

target_count = 10000

def get_data_frequency_stat(data, output_csv="frequency_stat.csv"):
    labels = data['labels']
    data = data['data'] # (32, 32, 3, 168233)
    mappings = load_mapping()
    unique_elements, counts = np.unique(labels, return_counts=True)
    map_symbols = np.vectorize(lambda x: mappings.get(str(x)).get("symbol"))
    symbols = map_symbols(unique_elements)
    df = pd.DataFrame({
        "label": unique_elements,
        "Symbol": symbols,
        "Frequency": counts
    })
    df.to_csv(output_csv, index=False)
    print(f"数据已保存到 {output_csv}")

def save_original_data_to_pt(data):
    labels = data['labels']
    data = data['data']
    # 将 data 转换为 PyTorch 张量，并调整形状为 (168233, 3, 32, 32)
    data = torch.tensor(data, dtype=torch.float32)  # 转换为张量
    data = data.permute(3, 2, 0, 1)  # 从 (32, 32, 3, 168233) 调整为 (168233, 3, 32, 32)

    # 将 labels 转换为 PyTorch 张量（如果不是的话）
    labels = torch.tensor(labels) if not isinstance(labels, torch.Tensor) else labels

    # 将数据和标签保存到一个 .pt 文件中
    save_path = "../data/HASYv2.pt"
    torch.save({
        'data': data,
        'labels': labels
    }, save_path)

    print(f"原始数据已保存到 {save_path}")

if __name__ == '__main__':
    HASYv2 = unpickle("../data/HASYv2")

    labels = HASYv2['labels']
    data = HASYv2['data'] # (32, 32, 3, 168233)

    data = np.transpose(data, (3, 2, 0, 1))  # 变为 (168233, 3, 32, 32) 适配 PyTorch

    save_dir = '../data/augmented_data/'
    os.makedirs(save_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.RandomAffine(
            degrees=15,
            translate=(0.1, 0.1),
            scale=(0.5, 1),
            # interpolation=InterpolationMode.BICUBIC,  # 双线性插值
            fill=(255, 255, 255)  # 填充背景色
        ),
        transforms.ToTensor()
    ])

    # 将数据按标签分组
    unique_labels = np.unique(labels)
    label_to_data = {label: [] for label in unique_labels}
    for idx, label in enumerate(labels):
        label_to_data[label[0]].append(data[idx])

    for label in unique_labels:
        original_data = label_to_data[label]
        original_count = len(original_data)
        augmented_data = []
        augmented_labels = []

        print(f"Processing label {label}, original count: {original_count}")

        # 先添加原始数据
        for img_data in original_data:
            augmented_data.append(torch.from_numpy(img_data))
            augmented_labels.append(label)

        # 如果原始数据不足 target_count，则进行增强
        if original_count < target_count:
            cycles = (target_count - original_count) // original_count + 1
            for _ in range(cycles):
                for img_data in original_data:
                    if len(augmented_data) >= target_count:
                        break
                    # 转换为 PIL 图像进行增强
                    img = Image.fromarray((img_data.transpose(1, 2, 0) * 255).astype(np.uint8))
                    augmented_img = transform(img)  # 已经是 tensor 格式
                    augmented_data.append(augmented_img)
                    augmented_labels.append(label)

        # 裁剪到确切的 target_count
        augmented_data = augmented_data[:target_count]
        augmented_labels = augmented_labels[:target_count]

        # 转换为 tensor
        augmented_data_tensor = torch.stack(augmented_data)
        augmented_labels_tensor = torch.tensor(augmented_labels)

        # 保存为 PyTorch 格式
        save_path = os.path.join(save_dir, f'label_{label}_augmented.pt')
        torch.save({
            'data': augmented_data_tensor,
            'labels': augmented_labels_tensor
        }, save_path)

        print(f"Saved {len(augmented_data)} samples for label {label} to {save_path}")

    print("Data augmentation completed!")
