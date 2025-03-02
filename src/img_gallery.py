import sys
import torch
import json
import base64
import numpy as np
from cnn import unpickle
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QGridLayout, QScrollArea
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtCore import Qt
import numpy as np


# Load dataset
# loaded = torch.load('../data/augmented_data/train_1_set.pt')
# data = loaded['data'][0:1000]  # shape: [n, 3, 32, 32]
# labels = loaded['labels']
start = 1000
end = 2000
HASYv2 = torch.load('../data/HASYv2.pt')
data = HASYv2['data'][start : end]  # shape: [n, 3, 32, 32]
labels = HASYv2['labels'][start : end]

# Load mappings
with open('../mappings/mappings.json', 'r') as f:
    mappings = json.load(f)

def tensor_to_qpixmap(tensor):
    """Convert a (3, 32, 32) tensor to QPixmap"""
    array = tensor.numpy().transpose(1, 2, 0)  # (H, W, C)
    array = (array * 255).astype(np.uint8)
    height, width, channel = array.shape
    qimage = QImage(array.tobytes(), width, height, channel * width, QImage.Format_RGB888)
    return QPixmap.fromImage(qimage)

class ImageGrid(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Browser")
        layout = QVBoxLayout()

        # Scroll Area
        scroll = QScrollArea()
        container = QWidget()
        grid = QGridLayout()

        self.image_buttons = []
        for i in range(len(data)):
            pixmap = tensor_to_qpixmap(data[i])
            btn = QPushButton()
            btn.setIcon(QIcon(pixmap))
            btn.setIconSize(pixmap.size())
            btn.clicked.connect(lambda checked, idx=i: self.open_details(idx))
            self.image_buttons.append(btn)
            grid.addWidget(btn, i // 4, i % 4)

        container.setLayout(grid)
        scroll.setWidget(container)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        self.setLayout(layout)

    def open_details(self, idx):
        self.detail_window = ImageDetail(idx)
        self.detail_window.show()

class ImageDetail(QWidget):
    def __init__(self, index):
        super().__init__()
        self.index = index
        self.setWindowTitle("Image Details")

        layout = QVBoxLayout()

        # Image
        self.image_label = QLabel()
        pixmap = tensor_to_qpixmap(data[self.index])
        self.image_label.setPixmap(pixmap)
        layout.addWidget(self.image_label)

        # Label Info
        label_id = str(labels[self.index].item())
        info = mappings.get(label_id, {"symbol": "?", "unicode": "?", "svg": ""})
        self.info_label = QLabel(f"Label: {label_id}\n\
Symbol: {info['symbol']}\n\
Unicode: {info['unicode']}\n\
Index: {str(self.index)}")
        layout.addWidget(self.info_label)

        # SVG Display
        svg_data = base64.b64decode(info.get("svg"))
        self.svg_widget = QSvgWidget()
        self.svg_widget.load(svg_data)
        self.svg_widget.setFixedSize(100, 100)
        layout.addWidget(self.svg_widget)

        # Navigation Buttons
        self.prev_btn = QPushButton("Previous")
        self.next_btn = QPushButton("Next")
        self.back_btn = QPushButton("Back")

        self.prev_btn.clicked.connect(self.show_previous)
        self.next_btn.clicked.connect(self.show_next)
        self.back_btn.clicked.connect(self.close)

        layout.addWidget(self.prev_btn)
        layout.addWidget(self.next_btn)
        layout.addWidget(self.back_btn)

        self.setLayout(layout)

    def show_previous(self):
        if self.index > 0:
            self.index -= 1
            self.update_details()

    def show_next(self):
        if self.index < len(data) - 1:
            self.index += 1
            self.update_details()

    def update_details(self):
        pixmap = tensor_to_qpixmap(data[self.index])
        self.image_label.setPixmap(pixmap)
        label_id = str(labels[self.index].item())
        info = mappings.get(label_id, {"symbol": "?", "unicode": "?", "svg": ""})
        self.info_label.setText(f"Label: {label_id}\nSymbol: {info['symbol']}\nUnicode: {info['unicode']}")

        if info["svg"]:
            self.svg_widget.load(base64.b64decode(info.get("svg")))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageGrid()
    window.show()
    sys.exit(app.exec_())
