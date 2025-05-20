"""模型评估模块"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from src.config import CLASSES

def evaluate_model(model, test_loader, device, model_type):
    """评估模型在测试集上的性能"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算评估指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted'
    )
    
    # 打印评估结果
    print(f"\n--- {model_type.upper()} 模型评估结果 ---")
    print(f"准确率 (Accuracy): {accuracy:.8f}")
    print(f"精确率 (Precision): {precision:.8f}")
    print(f"召回率 (Recall): {recall:.8f}")
    print(f"F1分数 (F1 Score): {f1:.8f}")
    
    # 返回指标结果
    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    
    
    return results