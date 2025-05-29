import os
import matplotlib.pyplot as plt
from src.models.cnn_model import cnnNet
from src.models.liquid_cnn_model import LiquidCNN
import torch
from torchvision import transforms
from PIL import Image


def prepare_input_image(image_path, input_size=(32, 32)):

    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose(
        [
            transforms.Resize(input_size),  # 调整图像大小
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # 归一化
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

    # 绘制特征图
    fig, axes = plt.subplots(len(features), 1, figsize=(10, 5 * len(features)))
    if len(features) == 1:
        axes = [axes]  # 如果只有一层，将axes转换为列表

    for i, feature_map in enumerate(features):
        num_features = feature_map.shape[1]  # 获取特征图数量
        num_columns = 8  # 每行显示的特征图数量
        num_rows = (num_features + num_columns - 1) // num_columns  # 计算行数

        # 创建子图
        subfig, subaxes = plt.subplots(num_rows, num_columns, figsize=(20, 20))
        if num_rows == 1:
            subaxes = [subaxes]  # 如果只有一行，将subaxes转换为列表

        for j in range(num_features):
            ax = subaxes[j // num_columns][j % num_columns]
            ax.imshow(feature_map[0, j].detach().cpu().numpy(), cmap="viridis")
            ax.axis("off")

        subfig.suptitle(f"Layer {i+1} Feature Maps")

        subfig.savefig(os.path.join("output/", f"feature_maps_layer_{i+1}.png"))
        plt.close(subfig)

    # 绘制所有层的特征图
    for i, feature_map in enumerate(features):
        axes[i].imshow(feature_map[0, 0].detach().cpu().numpy(), cmap="viridis")
        axes[i].set_title(f"Layer {i+1}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join("output/", "all_feature_maps.png"))
    plt.show()


# 加载模型和输入图像
cnn_model = cnnNet(num_classes=29)
cnn_model.load_state_dict(torch.load("output/cnn_training_model.pth"))
cnn_model.eval()

liquid_cnn_model = LiquidCNN(input_shape=(3, 32, 32), n_classes=29)
liquid_cnn_model.load_state_dict(torch.load("liquid_cnn_model.pth"))
liquid_cnn_model.eval()

# 准备输入图像
image_path = "path_to_your_image.jpg"  # 替换为你的图像路径
input_image = prepare_input_image(image_path, input_size=(32, 32))

# 可视化CNN特征图
visualize_feature_maps(cnn_model, input_image, model_type="CNN")

# 可视化Liquid CNN特征图
visualize_feature_maps(liquid_cnn_model, input_image, model_type="LiquidCNN")
