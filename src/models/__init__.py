"""模型模块"""

from models.cnn import cnnNet
from models.liquid_cnn import LiquidCNN


def get_model(model_type, num_classes, device, input_shape=(3, 32, 32)):
    """获取指定类型的模型"""
    if model_type.lower() == "cnn":
        return cnnNet(num_classes=num_classes).to(device)
    elif model_type.lower() == "liquid":
        return LiquidCNN(input_shape=input_shape, n_classes=num_classes).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'cnn' or 'liquid'")
