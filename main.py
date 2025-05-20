import torch
import numpy as np
import argparse

from src.config import TRAIN_DIR, INPUT_SHAPE, CPU_THREADS
from src.utils.data_processing import get_data, create_data_loaders
from src.models import get_model
from src.train.train import train_standard_model, train_liquid_model
from src.utils.visualize import plot_sample_images, plot_training_history
from src.evaluate import evaluate_model

def main():
    # 命令行参数
    parser = argparse.ArgumentParser(description="ASL Recognition with PyTorch")
    parser.add_argument(
        "--model",
        type=str,
        default="cnn",
        choices=["cnn", "liquid"],
        help="Model type: cnn or liquid (default: cnn)",
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
    if args.cpu_only:
        device = torch.device("cpu")
        if args.model.lower() == "liquid":
            # 为液态网络优化CPU
            torch.set_num_threads(CPU_THREADS)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Selected model: {args.model}")

    # 加载数据
    print("Loading data...")
    X, y = get_data(TRAIN_DIR)
    num_classes = len(np.unique(y))
    print(f"Loaded {len(X)} images with {num_classes} classes")

    # 可视化样本（如果指定）
    if args.visualize_samples:
        plot_sample_images()

    # 根据模型类型选择不同的训练流程
    if args.model.lower() == "cnn":
        # 标准CNN模型训练流程
        train_loader, val_loader, test_loader = create_data_loaders(X, y, with_validation=True)
        model = get_model("cnn", num_classes, device)
        print(model)

        # 训练模型
        trained_model, history = train_standard_model(
            model, train_loader, val_loader, device
        )

        # 保存模型
        torch.save(trained_model.state_dict(), "asl_cnn_model.pth")
        print("Model saved to asl_cnn_model.pth")

        # 评估模型
        results = evaluate_model(trained_model, test_loader, device, "cnn")

    elif args.model.lower() == "liquid":
        # 液态神经网络训练流程
        train_loader, test_loader = create_data_loaders(X, y, with_validation=False)
        model = get_model("liquid", num_classes, device, INPUT_SHAPE)
        print(model)

        # 训练模型
        trained_model, history = train_liquid_model(model, train_loader, device)

        # 保存模型
        torch.save(trained_model.state_dict(), "output/asl_liquid_model.pth")
        print("Model saved to output/asl_liquid_model.pth")

        # 评估模型
        results = evaluate_model(trained_model, test_loader, device, "liquid")

    # 可视化训练历史
    plot_training_history(history, args.model)


if __name__ == "__main__":
    main()
