import matplotlib.pyplot as plt
from pprint import pprint
import numpy as np

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

n = 120600
image = data[:, :, :, n]  # Shape (32, 32, 3)

# image_label = labels[n]
print(image)
print(symbols[n])
# Display the image
plt.imshow(image)
# plt.axis("off")  # Hide the axes
plt.show()

data = data[:, :, 0, :]
print(data.shape)
print(labels)
