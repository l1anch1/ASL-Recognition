"""液态神经网络模型"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torchdyn.models import NeuralODE

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from src.config import LIQUID_SOLVER


class LiquidCNN(nn.Module):
    """液态卷积神经网络，基于Neural ODE实现"""

    def __init__(self, input_shape=(3, 32, 32), n_classes=29):
        super().__init__()

        # CNN特征提取
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(128),
            nn.Dropout(0.2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(256),
        )

        # 动态计算特征维度
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            self.feature_dim = np.prod(self.features(dummy).shape)

        # 液态动态层
        self.dynamics = NeuralODE(
            nn.Sequential(
                nn.Linear(self.feature_dim, 4096),
                nn.Tanh(),
            ),
            solver=LIQUID_SOLVER,
            return_t_eval=False,
        )

        # 分类头
        self.classifier = nn.Linear(4096, n_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.dynamics(x)[-1]  # 取最终状态
        return self.classifier(x)


