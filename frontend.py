import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import base64
from PyQt5.QtWidgets import QListWidgetItem, QListWidget, QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel, QHBoxLayout
from PyQt5.QtGui import QPainter, QPen, QImage, QPixmap
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtSvg import QSvgWidget
from torchvision import transforms
from PIL import Image
from cnn import SimpleCNN, EfficientNetB0Model

class Canvas(QWidget):
    def __init__(self):
        super().__init__()
        self.scale_factor = 10
        self.setFixedSize(32 * self.scale_factor, 32 * self.scale_factor)
        self.image = QImage(32, 32, QImage.Format_RGB32)
        self.image.fill(Qt.white)
        self.last_point = None

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.scale(self.scale_factor, self.scale_factor)
        painter.drawImage(0, 0, self.image)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.last_point = self._map_to_image(event.pos())

    def mouseMoveEvent(self, event):
        if self.last_point is not None:
            painter = QPainter(self.image)
            pen = QPen(Qt.black, 2, Qt.SolidLine)
            painter.setPen(pen)
            current_point = self._map_to_image(event.pos())
            painter.drawLine(self.last_point, current_point)
            self.last_point = current_point
            self.update()

    def mouseReleaseEvent(self, event):
        self.last_point = None

    def _map_to_image(self, pos):
        return QPoint(pos.x() // self.scale_factor, pos.y() // self.scale_factor)

    def clear_canvas(self):
        self.image.fill(Qt.white)
        self.update()

    def save_image(self, img_path):
        self.image.save(img_path)

    def get_image(self, img_path):
        # image = Image.open(img_path).convert("L")
        image = Image.open(img_path)
        transform = transforms.ToTensor()
        img_tensor = transform(image)
        img_tensor = img_tensor.unsqueeze(0)
        return img_tensor


class MainWindow(QMainWindow):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.canvas = Canvas()
        self.setCentralWidget(self.canvas)

        # Prediction and Clear Buttons
        self.predict_button = QPushButton("predict")
        self.predict_button.clicked.connect(self.predict)

        self.clear_button = QPushButton("clear")
        self.clear_button.clicked.connect(self.canvas.clear_canvas)

        # Preview list for SVG images
        self.preview_list = QListWidget()

        # Main layout with preview on the right
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.predict_button)
        layout.addWidget(self.clear_button)

        container = QWidget()
        container.setLayout(layout)

        # Horizontal layout to hold the main widget and preview list
        main_layout = QHBoxLayout()
        main_layout.addWidget(container)
        main_layout.addWidget(self.preview_list)  # Add the preview list to the right

        main_container = QWidget()
        main_container.setLayout(main_layout)
        self.setCentralWidget(main_container)

        self.setWindowTitle("LaTeX symbol recognize")

    def predict(self):
        img_path = './imgs/test.jpg'
        mapping = load_mapping()
        self.canvas.save_image(img_path)
        img_tensor = self.canvas.get_image(img_path)

        # Perform prediction and get top 5 results
        with torch.no_grad():
            output = self.model(img_tensor)
            _, predicted = torch.topk(output, 5, dim=1)

        # Clear previous items in the preview list
        self.preview_list.clear()

        # Display top 5 results with SVGs
        for idx in predicted[0]:
            label = str(idx.item())
            if label in mapping:
                # svg widget
                svg_data = base64.b64decode(mapping.get(label, "")["svg"])
                svg_widget = QSvgWidget()
                svg_widget.load(svg_data)
                svg_widget.setFixedSize(100, 100)  # Set square size for SVG

                # LaTeX symbol discription
                label_label = QLabel()
                label_label.setText("label: " + label)

                symbol_label = QLabel()
                symbol_text = mapping.get(label, "")["symbol"]
                symbol_label.setText("symbol: " + symbol_text)

                unicode_label = QLabel()
                unicode_text = mapping.get(label, "")["unicode"]
                unicode_label.setText("unicode: " + unicode_text)

                discription_layout = QVBoxLayout()
                discription_layout.addWidget(label_label)
                discription_layout.addWidget(symbol_label)
                discription_layout.addWidget(unicode_label)

                discription_container = QWidget()
                discription_container.setLayout(discription_layout)

                # Horizontal layout to place SVG and LaTeX side by side
                item_layout = QHBoxLayout()
                item_layout.addWidget(svg_widget)
                item_layout.addWidget(discription_container)

                # Container widget to hold the layout
                item_container = QWidget()
                item_container.setLayout(item_layout)

                # Add container to QListWidget as a new item
                list_item = QListWidgetItem()
                list_item.setSizeHint(item_container.sizeHint())
                self.preview_list.addItem(list_item)
                self.preview_list.setItemWidget(list_item, item_container)


def load_model(model_path):
    # model = SimpleCNN()
    model = EfficientNetB0Model(370)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def load_mapping():
    with open("./mappings/mappings.json", "r", encoding="utf-8") as file:
        return json.load(file)

if __name__ == '__main__':
    model_path = './models/EfficientNetB0-epoch10.pth'
    app = QApplication(sys.argv)
    model = load_model(model_path)
    window = MainWindow(model)
    window.show()
    sys.exit(app.exec_())
