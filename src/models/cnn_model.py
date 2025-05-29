"""模型定义模块"""

import torch
import torch.nn as nn
import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from src.config import NUM_CLASSES, CNN_DROPOUT, CNN_KERNAL_SIZE


class cnnNet(nn.Module):
    """ASL手语识别CNN模型"""

    def __init__(self, num_classes=NUM_CLASSES):
        super(cnnNet, self).__init__()

        # Layer 1
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=CNN_KERNAL_SIZE, padding=CNN_KERNAL_SIZE // 2
        )
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.bn1 = nn.BatchNorm2d(64)

        # Layer 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.dropout1 = nn.Dropout(CNN_DROPOUT)

        # Layer 3
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.bn3 = nn.BatchNorm2d(256)

        # 使用自适应池化确保固定输出大小
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Fully connected layers
        self.flatten = nn.Flatten()
        self.dropout2 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(256 * 4 * 4, 1024)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # Layer 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.bn1(x)

        # Layer 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.dropout1(x)

        # Layer 3
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.bn3(x)

        # 确保输出大小为4x4
        x = self.adaptive_pool(x)

        # Fully connected layers
        x = self.flatten(x)
        x = self.dropout2(x)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)

        return x

    def forward_features(self, x):
        features = []
        # 层1
        x = self.conv1(x)
        features.append(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.bn1(x)
        # 层2
        x = self.conv2(x)
        features.append(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.dropout1(x)
        # 层3
        x = self.conv3(x)
        features.append(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.bn3(x)
        # 自适应池化
        x = self.adaptive_pool(x)
        features.append(x)
        return features
