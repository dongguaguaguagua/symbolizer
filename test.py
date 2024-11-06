import matplotlib.pyplot as plt
from pprint import pprint
import numpy as np
import json
from collections import OrderedDict
# np.set_printoptions(threshold=np.inf)

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

print("loading data...")
HASYv2 = unpickle("./data/HASYv2")
data = HASYv2['data']
labels = HASYv2['labels']
symbols = HASYv2['latex_symbol']
print("data loaded")

def print_img(n):
    image = data[:, :, :, n]
    print(symbols[n])
    plt.imshow(image)
    plt.axis("off")
    plt.show()

for n in range(168233):
    if(labels[n][0]==300):
        print(n)
        print_img(n+1)
        break

# unique_symbols = list(OrderedDict.fromkeys(str(symbol.item()) for symbol in symbols))
# print(unique_symbols)
