import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import numpy as np
from PIL import Image
import os
from cnn import unpickle
import json
from frontend import *
import pandas as pd

HASYv2 = unpickle("./data/HASYv2")
# labels = HASYv2['labels']
# data = HASYv2['data'] # (32, 32, 3, 168233)

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


# data = np.transpose(data, (3, 2, 0, 1))  # 变为 (168233, 3, 32, 32) 适配 PyTorch

# save_dir = './augmented_data/'
# os.makedirs(save_dir, exist_ok=True)

# transform = transforms.Compose([
#     transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.6, 1), fill=(255,255,255)), # 随机旋转、平移、缩放
# ])

# for i, img_data in enumerate(data):
#     img = Image.fromarray((img_data.transpose(1, 2, 0) * 255).astype(np.uint8))  # 恢复为 (32, 32, 3)
#     img.save(os.path.join(save_dir, f"{i}.png"))
#     augmented_img = transform(img)
#     augmented_img.save(os.path.join(save_dir, f"{i}_augmented.png"))

#     if i >= 99:
#         break

# print("数据增强并保存完成。")
