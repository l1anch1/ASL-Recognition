import os
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
from PIL import Image
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.cnn_model import cnnNet
from src.models.liquid_cnn_model import LiquidCNN


def prepare_input_image(image_path, input_size=(32, 32)):

    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose(
        [
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    input_tensor = transform(image)
    input_tensor = input_tensor.unsqueeze(0)

    return input_tensor


def visualize_feature_maps(model, input_image, model_type="CNN"):
    """
    通用特征图可视化函数，支持CNN和Liquid CNN。

    参数:
    - model: 模型对象
    - input_image: 输入图像张量
    - model_type: 模型类型，"CNN" 或 "LiquidCNN"
    """
    if model_type == "CNN":
        features = model.forward_features(input_image)
    elif model_type == "LiquidCNN":
        features = model.forward_features(input_image)
    else:
        raise ValueError("Unsupported model type. Choose 'CNN' or 'LiquidCNN'.")

    # 选择每层的特征图
    selected_features = []
    for feature_map in features:
        if feature_map.dim() == 4:  # 卷积层输出 (batch_size, channels, height, width)
            selected_features.append(feature_map[0, 0])  # 选择第一张图片的第一个通道
        elif feature_map.dim() == 2:  # 动态层输出 (batch_size, feature_dim)
            selected_features.append(feature_map.view(-1))  # 将其视为一维特征图
        else:
            print(f"Invalid feature map shape: {feature_map.shape}")

    total_features = len(selected_features)  # 选取的特征图数量
    num_columns = total_features  # 每行展示的特征图数量

    fig, axes = plt.subplots(1, num_columns, figsize=(20, 5))  # 水平排列
    if total_features == 1:
        axes = [axes]  # 确保 axes 是列表

    for i, feature_map in enumerate(selected_features):
        # 确保是 2D 数据适合 `imshow`
        if feature_map.dim() == 1:  # 一维特征图
            axes[i].imshow(
                feature_map.detach().cpu().numpy().reshape(1, -1),
                cmap="viridis",
                aspect="auto",
            )  # Reshape to 2D
        else:  # 处理多通道数据
            axes[i].imshow(feature_map.detach().cpu().numpy(), cmap="viridis")
        axes[i].axis("off")
        axes[i].set_title(f"Layer {i+1}")  # 可选择为每层添加标题

    plt.suptitle(f"{model_type} Feature Maps")
    plt.tight_layout()
    plt.savefig(os.path.join("output/", f"{model_type}_feature_maps.png"))
    plt.show()


def generate_feature_map_images(model_type):
    if model_type == "CNN":
        model = cnnNet(num_classes=29)
        model.load_state_dict(torch.load("output/asl_cnn_model.pth"))
        model.eval()
    elif model_type == "LiquidCNN":
        model = LiquidCNN(input_shape=(3, 32, 32), n_classes=29)
        model.load_state_dict(torch.load("output/asl_liquid_model.pth"))
        model.eval()

    image_path = "dataset/asl_alphabet_test/asl_alphabet_test/C_test.jpg"
    input_image = prepare_input_image(image_path, input_size=(32, 32))

    visualize_feature_maps(model, input_image, model_type=model_type)


generate_feature_map_images("CNN")
