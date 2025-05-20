"""训练函数模块"""

import torch
import torch.nn as nn
import torch.optim as optim

from src.config import EPOCHS, CNN_LEARNING_RATE, LIQUID_LEARNING_RATE


def train_standard_model(model, train_loader, val_loader, device):
    """训练标准CNN模型"""
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CNN_LEARNING_RATE)

    # 训练历史记录
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # 训练循环
    for epoch in range(EPOCHS):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 清零梯度
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels.long())

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # 统计数据
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # 计算训练指标
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

        # 计算验证指标
        val_loss = val_running_loss / len(val_loader)
        val_acc = 100.0 * val_correct / val_total

        # 保存历史记录
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch+1}/{EPOCHS}, "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

    return model, history


def train_liquid_model(model, train_loader, device):
    """训练液态神经网络模型"""
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LIQUID_LEARNING_RATE)

    # 训练历史
    history = {"train_loss": [], "train_acc": []}

    # 训练循环
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            # 将数据移动到设备
            images, labels = images.to(device), labels.to(device)

            # 前向传播与反向传播
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

            # 统计
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # 计算epoch平均损失和准确率
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total

        # 保存历史
        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc)

        # 打印统计信息
        print(
            f"Epoch {epoch + 1}/{EPOCHS}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.2f}%"
        )

    return model, history
