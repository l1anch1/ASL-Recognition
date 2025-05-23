# 设置设备
import os
import sys
import torch
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
from src.config import CLASSES
from src.utils.visualize import visualize_vit_attention
from src.models.vit_model import ViTNet  # 替换为你的模型模块导入

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

weight_path = "output/asl_vit_model.pth"

if not os.path.exists(weight_path):
	raise FileNotFoundError(f"权重文件 {weight_path} 不存在!")


model = ViTNet() 

# 加载权重
state_dict = torch.load(weight_path, map_location=device)
model.load_state_dict(state_dict)
print(f"成功加载权重: {weight_path}")

# 设置模型为评估模式
model.eval()


class_names = CLASSES

# 测试图像路径
image_path = "dataset/asl_alphabet_train/asl_alphabet_train/A/A8.jpg"

# 调用可视化函数
visualize_vit_attention(model, image_path, class_names, device)