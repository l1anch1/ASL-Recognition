"""训练函数模块"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from src.config import (
    EPOCHS, CNN_LEARNING_RATE, LIQUID_LEARNING_RATE, VIT_LEARNING_RATE,
    USE_SCHEDULER, SCHEDULER_TYPE, WEIGHT_DECAY, EARLY_STOPPING,
    EARLY_STOPPING_PATIENCE, OPTIMIZER_TYPE
)


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
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay_value)
    elif OPTIMIZER_TYPE.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay_value)
    elif OPTIMIZER_TYPE.lower() == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay_value)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay_value)
    
    # 配置学习率调度器
    scheduler = None
    if USE_SCHEDULER:
        if SCHEDULER_TYPE == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
        elif SCHEDULER_TYPE == "plateau":
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    
    # 初始化早停变量
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
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
        
        # 学习率调度器步进
        if scheduler is not None:
            if SCHEDULER_TYPE == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Current learning rate: {current_lr:.2e}")
        
        # 打印进度
        print(
            f"Epoch {epoch+1}/{EPOCHS}, "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )
        
        # 早停检查
        if EARLY_STOPPING:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型状态
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs")
                # 恢复到最佳模型状态
                if best_model_state:
                    model.load_state_dict(best_model_state)
                break
    
    # 如果使用了早停且有最佳模型状态，确保返回的是最佳模型
    if EARLY_STOPPING and best_model_state and patience_counter >= EARLY_STOPPING_PATIENCE:
        model.load_state_dict(best_model_state)
    
    return model, history