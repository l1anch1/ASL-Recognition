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
    
    # 计算精确率、召回率和F1分数的宏、微和加权平均
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro'
    )
    
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='micro'
    )

    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted'
    )
    
    # 打印评估结果
    print(f"\n--- {model_type.upper()} 模型评估结果 ---")
    print(f"准确率 (Accuracy): {accuracy:.8f}")
    print(f"宏平均精确率 (Macro Precision): {precision_macro:.8f}")
    print(f"宏平均召回率 (Macro Recall): {recall_macro:.8f}")
    print(f"宏平均F1分数 (Macro F1 Score): {f1_macro:.8f}")
    
    print(f"微平均精确率 (Micro Precision): {precision_micro:.8f}")
    print(f"微平均召回率 (Micro Recall): {recall_micro:.8f}")
    print(f"微平均F1分数 (Micro F1 Score): {f1_micro:.8f}")
    
    print(f"加权平均精确率 (Weighted Precision): {precision_weighted:.8f}")
    print(f"加权平均召回率 (Weighted Recall): {recall_weighted:.8f}")
    print(f"加权平均F1分数 (Weighted F1 Score): {f1_weighted:.8f}")
    
    # 返回指标结果
    results = {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_micro": f1_micro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted
    }
    
    return results