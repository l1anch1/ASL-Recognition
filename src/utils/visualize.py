"""可视化工具模块"""

import matplotlib.pyplot as plt
import os

from src.config import TRAIN_DIR, CLASSES


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
    plt.savefig(f"{model_type}_training_history.png")
    plt.show()
