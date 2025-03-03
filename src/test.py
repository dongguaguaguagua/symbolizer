import matplotlib.pyplot as plt
from pprint import pprint
import numpy as np
import json
import torch
from PIL import Image
from collections import OrderedDict
from common import load_data
from natsort import natsorted

# np.set_printoptions(threshold=np.inf)


def print_img(data, n):
    image = data[n]
    # print(symbols[n])
    image_np = image.permute(1, 2, 0).numpy() # 转化为NHWC顺序
    print(image_np)
    plt.imshow(image_np)
    plt.axis("off")
    plt.show()

def print_img_from_label(label):
    for n in range(168233):
        if labels[n][0] == label:
            print(n)
            print_img(n + 3)
            break


if __name__ == "__main__":
    print("loading data...")
    loaded = torch.load('../data/augmented_data/train_1_set.pt')
    data = loaded['data']      # shape: [100000, 3, 32, 32]
    labels = loaded['labels']  # shape: [100000]

    n = 939
    print(data[n][0][0][0] < 0.004 and data[n][0][0][0] > 0.0038)
    print(labels[n])
    print_img(data, n)
    # print_img_from_label(174)

# unique_symbols = list(OrderedDict.fromkeys(str(symbol.item()) for symbol in symbols))
# print(unique_symbols)
