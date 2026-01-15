import os
import torch
from torch.utils.data import TensorDataset, random_split
from data_augment import target_count
import random
from natsort import natsorted
import gc
import time

def split_and_merge_set(save_dir, output, size, iteration):
    all_sampled_data = []
    all_sampled_labels = []

    pt_files = [f for f in os.listdir(save_dir) if f.startswith("label_") and f.endswith("_augmented.pt")]
    pt_files = natsorted(pt_files)

    for pt_file in pt_files:
        file_path = os.path.join(save_dir, pt_file)

        print("加载.pt文件:", file_path)
        data_dict = torch.load(file_path)
        data, labels = data_dict['data'], data_dict['labels']

        start = iteration * size
        end = (iteration + 1) * size
        sampled_data = data[start:end]
        sampled_labels = labels[start:end]

        all_sampled_data.append(sampled_data.clone())  # 复制，避免引用原始data
        all_sampled_labels.append(sampled_labels.clone())
        # time.sleep(1)

    final_data = torch.cat(all_sampled_data, dim=0)
    final_labels = torch.cat(all_sampled_labels, dim=0)

    output_path = os.path.join(save_dir, output)
    torch.save({'data': final_data, 'labels': final_labels}, output_path)
    print(f"Merged set saved to {output_path}")

if __name__ == '__main__':
    save_dir = "../data/augmented_data/"
    target_parts = 20
    size = target_count // target_parts

    for i in range(target_parts):
        split_and_merge_set(save_dir, f'train_{i}_set.pt', size, i)
