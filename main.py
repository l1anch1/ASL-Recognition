import torch
import numpy as np
import argparse
import os

from src.config import TRAIN_DIR, CLASSES, INPUT_SHAPE, SAVE_DIR, TORCHINFO
from src.utils.data_processing import get_data, create_data_loaders
from src.models import get_model
from src.train import train_model
from src.utils.visualize import (
    plot_sample_images,
    plot_training_history,
)
from src.evaluate import evaluate_model
from src.utils.device_utils import get_device, print_device_info
from torchinfo import summary


def main():
    # 命令行参数
    parser = argparse.ArgumentParser(description="ASL Recognition with PyTorch")
    parser.add_argument(
        "--model",
        type=str,
        default="cnn",
        choices=["cnn", "liquid", "vit"],
        help="Model type: cnn, liquid or vit (default: cnn)",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Force CPU usage even if GPU is available",
    )
    parser.add_argument(
        "--visualize-samples",
        action="store_true",
        help="Visualize sample images before training",
    )
    args = parser.parse_args()

    # 设置设备
    device = get_device(cpu_only=args.cpu_only)
    print_device_info(device, args.model)

    # 加载数据
    print("Loading data...")
    X, y = get_data(TRAIN_DIR)
    num_classes = len(np.unique(y))
    print(f"Loaded {len(X)} images with {num_classes} classes")

    # 可视化样本（如果指定）
    if args.visualize_samples:
        plot_sample_images()

    # 模型配置
    model_configs = {
        "cnn": {
            "needs_input_shape": False,
            "with_validation": True,
            "save_path": "asl_cnn_model.pth",
        },
        "liquid": {
            "needs_input_shape": True,
            "with_validation": True,
            "save_path": "asl_liquid_model.pth",
        },
        "vit": {
            "needs_input_shape": False,
            "with_validation": True,
            "save_path": "asl_vit_model.pth",
        },
    }

    # 获取当前模型配置
    config = model_configs[args.model.lower()]

    # 创建数据加载器
    if config["with_validation"]:
        train_loader, val_loader, test_loader = create_data_loaders(
            X, y, with_validation=True
        )
    else:
        train_loader, test_loader = create_data_loaders(X, y, with_validation=False)
        val_loader = None

    # 创建模型
    if config["needs_input_shape"]:
        model = get_model(args.model, num_classes, device, INPUT_SHAPE)
    else:
        model = get_model(args.model, num_classes, device)
    print(model)

    # 打印模型结构
    if TORCHINFO:
        print("模型结构:")
        summary(model, input_size=(1, 3, 32, 32))

    # 训练模型
    trained_model, history = train_model(
        model, train_loader, val_loader, device, model_type=args.model
    )

    # 确保保存目录存在
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, config["save_path"])

    # 保存模型
    torch.save(trained_model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # 评估模型
    results = evaluate_model(trained_model, test_loader, device, args.model)

    # 可视化训练历史
    plot_training_history(history, args.model)


if __name__ == "__main__":
    main()
