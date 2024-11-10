import matplotlib.pyplot as plt
from pprint import pprint
import numpy as np
import json
from collections import OrderedDict
from common import load_data
from natsort import natsorted

# np.set_printoptions(threshold=np.inf)


def print_img(n):
    image = data[:, :, :, n]
    print(symbols[n])
    plt.imshow(image)
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
    data, labels = load_data("./data/HASYv2")
    print_img_from_label(174)

# unique_symbols = list(OrderedDict.fromkeys(str(symbol.item()) for symbol in symbols))
# print(unique_symbols)
