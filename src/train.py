"""训练函数模块"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import time
import functools

from src.config import (
    EPOCHS,
    CNN_LEARNING_RATE,
    LIQUID_LEARNING_RATE,
    VIT_LEARNING_RATE,
    USE_SCHEDULER,
    SCHEDULER_TYPE,
    WEIGHT_DECAY,
    EARLY_STOPPING,
    EARLY_STOPPING_PATIENCE,
    OPTIMIZER_TYPE,
)


def training_time_logger(func):
    """一个记录并打印函数执行时间的装饰器，并将时间添加到返回的 history 对象中。"""

    @functools.wraps(func)  # 保留原始函数的元数据
    def wrapper(*args, **kwargs):
        start_time = time.time()
        # 调用原始函数 (train_model)
        model, history = func(*args, **kwargs)
        end_time = time.time()

        total_training_time = end_time - start_time
        print(f"Total training time: {total_training_time:.2f} seconds")

        if isinstance(history, dict):
            history["total_training_time_seconds"] = total_training_time
        else:
            print(
                "Warning: Could not add training time to history as it's not a dictionary."
            )

        return model, history

    return wrapper


@training_time_logger
def train_model(model, train_loader, val_loader, device, model_type="cnn"):

    # 设置损失函数
    criterion = nn.CrossEntropyLoss()

    # 根据模型类型选择适当的学习率和优化器
    if model_type.lower() == "cnn":
        lr = CNN_LEARNING_RATE
        weight_decay_value = 0.0
    elif model_type.lower() == "liquid":
        lr = LIQUID_LEARNING_RATE
        weight_decay_value = 0.0
    elif model_type.lower() == "vit":
        lr = VIT_LEARNING_RATE
        weight_decay_value = WEIGHT_DECAY
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    # 根据配置选择优化器
    if OPTIMIZER_TYPE.lower() == "adam":
        optimizer = optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay_value
        )
    elif OPTIMIZER_TYPE.lower() == "sgd":
        optimizer = optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay_value
        )
    elif OPTIMIZER_TYPE.lower() == "adamw":
        optimizer = optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay_value
        )
    else:
        optimizer = optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay_value
        )  # 默认使用 Adam

    # 配置学习率调度器
    scheduler = None
    if USE_SCHEDULER:
        if SCHEDULER_TYPE == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
        elif SCHEDULER_TYPE == "plateau":
            scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3)

    # 初始化早停变量
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_time": [],
    }

    # 训练循环
    for epoch in range(EPOCHS):

        epoch_start_time = time.time()
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total

        # 验证阶段
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels.long())

                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_running_loss / len(val_loader)
        val_acc = 100.0 * val_correct / val_total

        # 计算本轮训练时间
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epoch_time"].append(epoch_time)

        if scheduler is not None:
            if SCHEDULER_TYPE == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Current learning rate: {current_lr:.2e}")

        print(
            f"Epoch {epoch+1}/{EPOCHS}, "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
            f"Time: {epoch_time:.2f}s"
        )

        if EARLY_STOPPING:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs")
                if best_model_state:
                    model.load_state_dict(best_model_state)
                break

    if (
        EARLY_STOPPING
        and best_model_state
        and patience_counter >= EARLY_STOPPING_PATIENCE
    ):
        model.load_state_dict(best_model_state)

    return model, history
