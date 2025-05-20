# 基于 PyTorch 的 ASL 手语识别
# DynamicHandSign: ASL Recognition with Neural ODEs

这个项目利用深度学习和PyTorch实现了美国手语（ASL）的识别。它提供了两种不同的神经网络架构：

标准卷积神经网络（CNN）
基于神经常微分方程（Neural ODE）的液态神经网络

## 功能特点
- 基于图像的ASL手语识别
- 两种模型架构：CNN和液态神经网络
- 完整的数据处理流水线
- 训练和评估功能
- 模型性能可视化工具
- 灵活的命令行界面


## 依赖库
```txt
matplotlib>=3.7.0
numpy>=1.20.0
opencv_contrib_python>=4.5.0
opencv_python>=4.5.0
scikit_learn>=1.0.0
torch>=2.0.0
torchdyn>=0.9.0
torchvision>=0.15.0
```
## 项目结构
```
ASL-Recognition
├─ docs
│  └─ model_structure.txt
├─ main.py
├─ output/
├─ README.md
├─ requirements.txt
└─ src
   ├─ config.py
   ├─ evaluate.py
   ├─ models
   │  ├─ cnn.py
   │  ├─ liquid_cnn.py
   │  └─ __init__.py
   ├─ train
   │  ├─ gpu_test.py
   │  └─ train.py
   └─ utils
      ├─ data_processing.py
      └─ visualize.py
```
## 模型架构

#### CNN模型
一个标准的卷积神经网络，架构如下：

- 3个卷积层（64、128、256个滤波器）
- 每个卷积层后接最大池化和批量归一化
- Dropout层用于正则化
- 全连接层用于分类

#### 液态神经网络
基于神经常微分方程的高级架构：

- CNN特征提取前端（类似于标准CNN）
- 基于神经ODE的动态层用于时间处理
- 线性分类头


## 使用方法
```Python
# 使用标准CNN模型训练（默认）
python main.py

# 使用液态神经网络训练
python main.py --model liquid

# 即使GPU可用也强制使用CPU
python main.py --cpu-only

# 在训练前可视化样本图像
python main.py --visualize-samples
```

## 命令行参数
- --model：选择模型架构（cnn或liquid）。默认：cnn
- --cpu-only：即使GPU可用也强制使用CPU
- --visualize-samples：在训练前显示数据集中的样本图像

## 训练过程
- 加载和预处理数据（调整大小为32x32像素并归一化）
- 根据指定架构初始化模型
- 使用适当的优化器和损失函数进行训练
- 在验证集上评估模型性能（对于标准CNN）
- 可视化训练指标并保存模型
- 在测试集上评估模型
## 模型输出
训练完成后，模型将被保存为：

- CNN模型保存为`asl_cnn_model.pth`
- 液态神经网络模型保存为`asl_liquid_model.pth`
## 性能可视化
训练脚本会自动生成训练指标的可视化图表：

- 训练和验证损失曲线
- 训练和验证准确率曲线

## 关于液态神经网络的说明
液态神经网络实现基于 torchdyn 库创建神经 ODE 模型。这种架构可能在时序数据上提供更好的性能，但在训练过程中需要更多的计算资源。

