import torch


print(f"CUDA 是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():

    x = torch.rand(5, 3)

    device = torch.device("cuda:0")
    x = x.to(device)
    print(f"张量在设备: {x.device}")
    print("GPU 测试成功!")
else:
    print("CUDA 不可用，请检查驱动和 PyTorch 安装")
