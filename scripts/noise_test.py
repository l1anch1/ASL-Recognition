import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from tqdm import tqdm
import cv2
import pandas as pd

from src.config import TRAIN_DIR, RANDOM_SEED, OUTPUT_DIR


# 设置随机种子，确保结果可复现
def set_seed(seed=RANDOM_SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed()

# 设备配置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------- 噪声生成函数 -----------------


def add_gaussian_noise(image, mean=0, std=0.1):
    """添加高斯噪声"""
    noise = torch.randn_like(image) * std + mean
    noisy_image = image + noise
    return torch.clamp(noisy_image, 0, 1)


def add_salt_pepper_noise(image, density=0.1):
    """添加椒盐噪声"""
    noisy_image = image.clone()
    h, w = image.shape[1], image.shape[2]

    # Salt noise
    num_salt = int(density * h * w * 0.5)
    y_coords = torch.randint(0, h, (num_salt,))
    x_coords = torch.randint(0, w, (num_salt,))
    noisy_image[:, y_coords, x_coords] = 1.0

    # Pepper noise
    num_pepper = int(density * h * w * 0.5)
    y_coords = torch.randint(0, h, (num_pepper,))
    x_coords = torch.randint(0, w, (num_pepper,))
    noisy_image[:, y_coords, x_coords] = 0.0

    return noisy_image


def add_blur(image, kernel_size=5):
    """添加高斯模糊"""
    # 将张量转换为numpy数组以应用cv2
    image_np = image.numpy().transpose(1, 2, 0)
    blurred = cv2.GaussianBlur(image_np, (kernel_size, kernel_size), 0)
    # 转回张量
    return torch.from_numpy(blurred.transpose(2, 0, 1))


# ----------------- 数据集定义 -----------------


class ASLTestDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        ASL测试数据集加载
        :param data_dir: 测试数据目录
        :param transform: 变换
        """
        self.data_dir = data_dir
        self.transform = transform
        self.classes = sorted(
            [
                d
                for d in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir, d))
            ]
        )
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self.samples = []
        for target_class in self.classes:
            class_dir = os.path.join(data_dir, target_class)
            for file_name in os.listdir(class_dir):
                if file_name.endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append(
                        (
                            os.path.join(class_dir, file_name),
                            self.class_to_idx[target_class],
                        )
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)

        return img, target


# ----------------- 抗噪性能评估函数 -----------------


def evaluate_noise_robustness(model, test_loader, noise_type, noise_levels, device):
    """
    评估模型在不同噪声级别下的性能
    :param model: 待评估模型
    :param test_loader: 测试数据加载器
    :param noise_type: 噪声类型 ('gaussian', 'salt_pepper', 'blur')
    :param noise_levels: 噪声级别列表
    :param device: 计算设备
    :return: 各噪声级别下的准确率
    """
    model.eval()
    results = []

    for noise_level in noise_levels:
        print(f"Evaluating {noise_type} noise at level {noise_level}")
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, targets in tqdm(test_loader):
                images = images.to(device)

                # 添加噪声
                if noise_type == "gaussian":
                    noisy_images = add_gaussian_noise(images, std=noise_level)
                elif noise_type == "salt_pepper":
                    noisy_images = torch.stack(
                        [
                            add_salt_pepper_noise(img, density=noise_level)
                            for img in images
                        ]
                    )
                elif noise_type == "blur":
                    noisy_images = torch.stack(
                        [add_blur(img, kernel_size=int(noise_level)) for img in images]
                    )

                # 模型推理
                outputs = model(noisy_images)
                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.numpy())

        # 计算评估指标
        accuracy = accuracy_score(all_targets, all_preds)
        precision = precision_score(all_targets, all_preds, average="macro")
        recall = recall_score(all_targets, all_preds, average="macro")
        f1 = f1_score(all_targets, all_preds, average="macro")

        results.append(
            {
                "noise_type": noise_type,
                "noise_level": noise_level,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

        print(f"Accuracy at {noise_type} noise level {noise_level}: {accuracy:.4f}")

    return results


def visualize_noise_examples(test_loader, noise_types, noise_levels):
    """
    可视化不同类型和级别的噪声效果
    """
    # 获取一个批次的图像
    images, _ = next(iter(test_loader))
    sample_img = images[0]

    fig, axs = plt.subplots(len(noise_types), len(noise_levels) + 1, figsize=(15, 10))

    # 显示原始图像
    for i, noise_type in enumerate(noise_types):
        axs[i, 0].imshow(sample_img.permute(1, 2, 0).numpy())
        axs[i, 0].set_title("Original")
        axs[i, 0].axis("off")

    # 显示添加噪声后的图像
    for i, noise_type in enumerate(noise_types):
        for j, level in enumerate(noise_levels[noise_type]):
            if noise_type == "gaussian":
                noisy_img = add_gaussian_noise(
                    sample_img.unsqueeze(0), std=level
                ).squeeze(0)
            elif noise_type == "salt_pepper":
                noisy_img = add_salt_pepper_noise(sample_img, density=level)
            elif noise_type == "blur":
                noisy_img = add_blur(sample_img, kernel_size=int(level))

            axs[i, j + 1].imshow(noisy_img.permute(1, 2, 0).numpy())
            axs[i, j + 1].set_title(f"{noise_type} {level}")
            axs[i, j + 1].axis("off")

    plt.tight_layout()
    plt.savefig("noise_examples.png")
    plt.show()


def plot_noise_comparison(results_dict, noise_type):
    """
    绘制不同模型在特定噪声类型下的性能比较图
    """
    plt.figure(figsize=(10, 6))

    for model_name, results in results_dict.items():
        # 过滤特定噪声类型的结果
        filtered_results = [r for r in results if r["noise_type"] == noise_type]

        # 提取噪声级别和准确率
        noise_levels = [r["noise_level"] for r in filtered_results]
        accuracies = [r["accuracy"] for r in filtered_results]

        plt.plot(noise_levels, accuracies, marker="o", label=model_name)

    plt.title(f"Model Robustness to {noise_type.capitalize()} Noise")
    plt.xlabel("Noise Level")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.savefig(f"{noise_type}_noise_comparison.png", dpi=300)
    plt.show()


# ----------------- 主函数 -----------------


def main():
    # 数据转换
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # 加载测试数据集
    test_dataset = ASLTestDataset(data_dir=TRAIN_DIR, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # 定义噪声类型和级别
    noise_types = ["gaussian", "salt_pepper", "blur"]
    noise_levels = {
        "gaussian": [0.01, 0.05, 0.10, 0.15, 0.25],
        "salt_pepper": [0.01, 0.05, 0.10, 0.15, 0.20],
        "blur": [1, 2, 3, 5, 7],  # 高斯模糊核大小
    }

    # 可视化噪声示例
    visualize_noise_examples(test_loader, noise_types, noise_levels)

    # 加载预训练模型
    models = {
        "CNN": "output/cnn_best.pth",
        "Liquid_CNN": "output/liquid_cnn_best.pth",
        "ViT": "output/vit_best.pth",
    }

    results = {}

    for model_name, model_path in models.items():
        print(f"\nEvaluating {model_name} model...")
        # 加载模型
        model = torch.load(model_path, map_location=device)
        model.to(device)
        model.eval()

        # 评估模型在各类噪声下的性能
        model_results = []
        for noise_type in noise_types:
            noise_results = evaluate_noise_robustness(
                model, test_loader, noise_type, noise_levels[noise_type], device
            )
            model_results.extend(noise_results)

        results[model_name] = model_results

    # 保存结果
    all_results = []
    for model_name, model_results in results.items():
        for result in model_results:
            result["model"] = model_name
            all_results.append(result)

    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_DIR + "noise_robustness_results.csv", index=False)

    # 绘制对比图
    for noise_type in noise_types:
        plot_noise_comparison(results, noise_type)

    # 生成详细报告
    print("\n===== 抗噪性能详细报告 =====")
    for noise_type in noise_types:
        print(f"\n{noise_type.capitalize()} Noise Performance:")
        for level in noise_levels[noise_type]:
            print(f"\nNoise Level: {level}")
            for model_name in models.keys():
                filtered_results = [
                    r
                    for r in results[model_name]
                    if r["noise_type"] == noise_type and r["noise_level"] == level
                ]
                if filtered_results:
                    acc = filtered_results[0]["accuracy"] * 100
                    print(f"{model_name}: {acc:.2f}%")


if __name__ == "__main__":
    main()

# 三张噪声类型对比图（保存为PNG格式）
# 一个包含所有结果的CSV文件
# 控制台输出的详细性能报告
