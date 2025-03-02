import os
import torch
from torch.utils.data import TensorDataset, random_split
from data_augument import target_count

def split_and_merge_set(save_dir, output, size):
    # 存储所有测试集数据
    all_test_data = []
    all_test_labels = []

    # 获取目录中所有.pt文件
    pt_files = [f for f in os.listdir(save_dir) if f.endswith('.pt')]

    for pt_file in pt_files:
        file_path = os.path.join(save_dir, pt_file)

        # 加载.pt文件
        data_dict = torch.load(file_path)
        data = data_dict['data']
        labels = data_dict['labels']

        # 创建数据集
        dataset = TensorDataset(data, labels)
        total_size = len(dataset)
        # size = int(total_size * 0.1)  # 10%作为测试集
        train_size = total_size - size

        # 随机分割
        train_dataset, test_dataset = random_split(dataset, [train_size, size])

        # 提取训练和测试数据
        train_data = torch.stack([x[0] for x in train_dataset])
        train_labels = torch.tensor([x[1] for x in train_dataset])
        test_data = torch.stack([x[0] for x in test_dataset])
        test_labels = torch.tensor([x[1] for x in test_dataset])

        # 将测试数据添加到总集合中
        all_test_data.append(test_data)
        all_test_labels.append(test_labels)

        # 保存更新后的训练数据到原文件
        torch.save({
            'data': train_data,
            'labels': train_labels
        }, file_path)

        # print(f"Processed {pt_file}: {size} samples moved to test set")

    # 合并所有测试数据
    merged_test_data = torch.cat(all_test_data, dim=0)
    merged_test_labels = torch.cat(all_test_labels, dim=0)

    # 保存合并后的测试集
    output_path = os.path.join(save_dir, output)
    torch.save({
        'data': merged_test_data,
        'labels': merged_test_labels
    }, output_path)

    print(f"Merged set saved to {output_path}")
    print(f"Total samples: {len(merged_test_data)}")

if __name__ == '__main__':
    save_dir = "../data/augmented_data/"  # 替换为你的保存目录
    target_parts = 10
    size = target_count // target_parts
    for i in range(1, target_parts):
        split_and_merge_set(save_dir, f'train_{i}_set.pt', size)


