"""通用数据加载和处理模块"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

from src.config import (
    TRAIN_DIR,
    BATCH_SIZE,
    TEST_SIZE,
    VAL_SIZE,
    RANDOM_SEED,
    NORM_MEAN,
    NORM_STD,
)


def get_data(data_dir):
    """加载图像数据并返回numpy数组"""
    images, labels = [], []

    # 获取目录列表并过滤掉隐藏文件和非目录项
    all_items = os.listdir(data_dir)
    class_dirs = [
        item
        for item in all_items
        if not item.startswith(".") and os.path.isdir(os.path.join(data_dir, item))
    ]
    class_dirs = sorted(class_dirs)  # 确保类别顺序一致

    for class_idx, class_name in enumerate(class_dirs):
        class_path = os.path.join(data_dir, class_name)
        print(f"Loading {class_name}...")

        # 同样过滤掉图像目录中的隐藏文件
        image_files = [img for img in os.listdir(class_path) if not img.startswith(".")]

        for img_name in image_files:
            img = cv2.imread(os.path.join(class_path, img_name))
            if img is None:  # 增加检查确保图像正确加载
                print(f"Warning: Could not load {img_name} in {class_name}")
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转为RGB
            img = cv2.resize(img, (32, 32))  # 调整尺寸
            images.append(img)
            labels.append(class_idx)

    return np.array(images), np.array(labels)


class ASLDataset(Dataset):
    """ASL手语数据集类"""

    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        else:
            # 默认转换为张量并归一化到[0,1]
            image = torch.from_numpy(image.transpose((2, 0, 1))).float() / 255.0
        return image, label


def create_data_loaders(X, y, with_validation=True):
    """创建训练、验证和测试数据加载器"""
    # 数据变换
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(NORM_MEAN, NORM_STD)]
    )

    # 首先分离出测试集
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )

    # 创建测试集数据加载器
    test_dataset = ASLDataset(X_test, y_test, transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    if with_validation:
        # 从剩余数据中拆分训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=VAL_SIZE,
            stratify=y_temp,
            random_state=RANDOM_SEED,
        )

        # 创建数据集
        train_dataset = ASLDataset(X_train, y_train, transform)
        val_dataset = ASLDataset(X_val, y_val, transform)

        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        return train_loader, val_loader, test_loader
    else:
        # 创建训练加载器（用于液态网络）
        train_dataset = ASLDataset(X_temp, y_temp, transform)
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        return train_loader, test_loader
