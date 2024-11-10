import torch
import torchvision
from cnn import SimpleCNN, EfficientNetB0Model

model_path = "./models/EfficientNetB0-epoch10.pth"
model = EfficientNetB0Model(370)
model.load_state_dict(torch.load(model_path))
model.eval()

dummy_input = torch.randn(1, 3, 32, 32)

torch.onnx.export(
    model,  # 要转换的模型
    dummy_input,  # 示例输入张量
    "./models/EfficientNetB0Model.onnx",  # ONNX 模型文件的保存路径
    export_params=True,  # 导出时包含模型参数
    opset_version=11,  # ONNX 的 opset 版本
    input_names=["input"],  # 输入节点名称
    output_names=["output"],  # 输出节点名称
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
    },  # 动态批量大小
)
