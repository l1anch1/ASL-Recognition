import os
import sys
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

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
from src.config import TRAIN_DIR, RANDOM_SEED, SAVE_DIR
from src.models.cnn_model import cnnNet
from src.models.liquid_cnn_model import LiquidCNN
from src.models.vit_model import ViTNet

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
    # 使用 cpu() 方法将张量移到 CPU
    image_np = image.cpu().numpy().transpose(1, 2, 0)
    blurred = cv2.GaussianBlur(image_np, (kernel_size, kernel_size), 0)
    # 转回张量
    return torch.from_numpy(blurred.transpose(2, 0, 1)).to(image.device)  # 传回 GPU


# ----------------- 数据集定义 -----------------


class ASLTestDataset(Dataset):
    def __init__(self, data_dir, transform=None, max_samples_per_class=500):
        """
        ASL测试数据集加载
        :param data_dir: 测试数据目录
        :param transform: 变换
        :param max_samples_per_class: 每个类的最大样本数
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

            all_images = [
                os.path.join(class_dir, file_name)
                for file_name in os.listdir(class_dir)
                if file_name.endswith((".jpg", ".jpeg", ".png"))
            ]

            limited_images = all_images[:max_samples_per_class]
            for image_path in limited_images:
                self.samples.append((image_path, self.class_to_idx[target_class]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)

        return img, target

def evaluate_noise_robustness(model, test_loader, noise_type, noise_levels, device):

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
	# 获取一个批次的图像
	images, _ = next(iter(test_loader))
	sample_img = images[0]

	num_noise_types = len(noise_types)
	num_noise_levels = max(len(noise_levels[noise_type]) for noise_type in noise_types)
	fig, axs = plt.subplots(num_noise_types, num_noise_levels + 1, figsize=(15, 10))

	# 显示原始图像
	for i, noise_type in enumerate(noise_types):
		axs[i, 0].imshow(sample_img.permute(1, 2, 0).numpy())
		axs[i, 0].set_title("Original")
		axs[i, 0].axis("off")

		# 显示添加噪声后的图像
		for j, level in enumerate(noise_levels[noise_type]):
			if j < num_noise_levels:  # 确保不会超出索引
				if noise_type == "gaussian":
					noisy_img = add_gaussian_noise(sample_img.unsqueeze(0), std=level).squeeze(0)
				elif noise_type == "salt_pepper":
					noisy_img = add_salt_pepper_noise(sample_img, density=level)
				elif noise_type == "blur":
					noisy_img = add_blur(sample_img, kernel_size=int(level))

				axs[i, j + 1].imshow(noisy_img.permute(1, 2, 0).numpy())
				axs[i, j + 1].set_title(f"{noise_type} {level}")
				axs[i, j + 1].axis("off")
	
	plt.tight_layout()
	plt.savefig(SAVE_DIR+"noise_examples.png")
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
	plt.savefig(SAVE_DIR+f"{noise_type}_noise_comparison.png", dpi=300)
	plt.show()

def main():
	# 数据转换
	transform = transforms.Compose(
		[
			transforms.ToPILImage(),
			transforms.Resize((32, 32)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
		]
	)

	# 加载测试数据集
	test_dataset = ASLTestDataset(data_dir=TRAIN_DIR, transform=transform)
	test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

	# 定义噪声类型和级别
	noise_types = ["gaussian", "salt_pepper", "blur"]
	noise_levels = {
		"gaussian": [0.01, 0.05, 0.10],
		"salt_pepper": [0.01, 0.05, 0.10],
		"blur": [1, 3, 5],  # 高斯模糊核大小
	}

	visualize_noise_examples(test_loader, noise_types, noise_levels)

	models = {
		"CNN": "output/asl_cnn_model.pth",
		"Liquid_CNN": "output/asl_liquid_model.pth",
		"ViT": "output/asl_vit_model.pth",
	}

	results = {}

	for model_name, model_path in models.items():
		print(f"\nEvaluating {model_name} model...")
		# 创建模型实例
		if model_name == "CNN":
			model = cnnNet()
		elif model_name == "Liquid_CNN":
			model = LiquidCNN()
		elif model_name == "ViT":
			model = ViTNet()
		

		state_dict = torch.load(model_path, map_location=device)
		model.load_state_dict(state_dict)  # 将权重加载到模型
		model.to(device)  # 将模型移动到设备
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
	df.to_csv(SAVE_DIR + "noise_robustness_results.csv", index=False)

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
