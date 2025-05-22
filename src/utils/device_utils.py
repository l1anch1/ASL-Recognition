import torch
def get_device(cpu_only=False):
    """自动检测设备，支持 Mac (MPS), CUDA 和 CPU"""
    if cpu_only:
        device = torch.device("cpu")
    else:
        # MPS is for Apple Silicon (M1/M2/M3) GPUs with PyTorch >= 1.12
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    return device


def print_device_info(device, model_type):
    """打印设备和模型信息"""
    print(f"Using device: {device}")
    if str(device) == "mps":
        print("Detected Apple Silicon (M1/M2/M3) - using MPS acceleration")
    elif str(device).startswith("cuda"):
        print("Using Nvidia GPU (CUDA)")
    else:
        print("Using CPU")
    print(f"Selected model: {model_type}")