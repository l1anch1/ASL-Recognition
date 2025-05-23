"""可视化工具模块"""

import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import cv2

from src.config import NORM_MEAN, NORM_STD, TRAIN_DIR, CLASSES, SAVE_DIR


def plot_sample_images():
    """显示每个类别的示例图像"""
    plt.figure(figsize=(16, 5))

    for i in range(len(CLASSES)):
        plt.subplot(3, 10, i + 1)
        plt.xticks([])
        plt.yticks([])
        path = os.path.join(TRAIN_DIR, CLASSES[i], f"{CLASSES[i]}1.jpg")
        img = plt.imread(path)
        plt.imshow(img)
        plt.xlabel(CLASSES[i])

    plt.tight_layout()
    plt.show()


def plot_training_history(history, model_type):
    """绘制训练历史"""
    plt.figure(figsize=(12, 6))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history["train_acc"], label="train_accuracy")
    if "val_acc" in history:
        plt.plot(history["val_acc"], label="val_accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title(f"{model_type} Model Accuracy")
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history["train_loss"], label="train_loss")
    if "val_loss" in history:
        plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"{model_type} Model Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(SAVE_DIR+f"{model_type}_training_history.png")
    plt.show()

def visualize_vit_attention(model, image_path, class_names=None, device="cuda"):
    """
    可视化Vision Transformer模型的注意力热图
    
    参数:
        model: 训练好的ViT模型
        image_path: 输入图像路径
        class_names: 类别名称列表，用于显示预测结果
        device: 使用的设备，默认"cuda"
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2
    from PIL import Image
    import torchvision.transforms as transforms
    
    # 图像预处理
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    # 获取原始图像用于叠加显示
    img_array = np.array(img.resize((32, 32))) / 255.0
    
    # 设置模型为评估模式
    model = model.to(device)
    model.eval()
    
    # 前向传播获取注意力权重和预测
    with torch.no_grad():
        outputs = model(input_tensor)
        # 获取注意力权重
        attention_weights = model.get_attention_weights()
        
        if class_names and len(class_names) > 0:
            _, predicted = torch.max(outputs, 1)
            predicted_class = class_names[predicted.item()]
            confidence = torch.nn.functional.softmax(outputs, dim=1)[0, predicted.item()].item()
    
    # 处理注意力权重形状 [batch, heads, seq_len, seq_len]
    # 平均所有注意力头
    attn_weights = attention_weights.mean(1)[0]  # [seq_len, seq_len]
    
    # [CLS] token对patch tokens的注意力
    cls_attention = attn_weights[0, 1:]  # 排除CLS对自身的注意力
    
    # 重塑为二维热图
    num_patches_side = int((model.img_size // model.patch_size))
    attention_map = cls_attention.reshape(num_patches_side, num_patches_side).cpu().numpy()
    
    # 上采样到原图大小
    attention_map = cv2.resize(attention_map, (32, 32))
    
    # 归一化热图值到[0,1]
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
    
    # 创建热图
    heatmap = cv2.applyColorMap(np.uint8(255 * attention_map), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    
    # 热图叠加到原图
    visualization = 0.7 * img_array + 0.3 * heatmap
    
    # 显示结果
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img_array)
    plt.title("Origin image")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(attention_map, cmap='jet')
    plt.title("attention map")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(visualization)
    if class_names and len(class_names) > 0:
        plt.title(f"Overlay Result\nPrediction: {predicted_class}\nConfidence: {confidence:.2f}")
    else:
        plt.title("Overlay Result")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(SAVE_DIR+"vit_attention_map.png")
    plt.show()
    
    return attention_map, visualization