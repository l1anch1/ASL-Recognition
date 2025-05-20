"""统一配置参数"""

# 数据路径
TRAIN_DIR = "dataset/asl_alphabet_train/asl_alphabet_train"
TEST_DIR = "dataset/asl_alphabet_test/asl_alphabet_test"

# 类别列表
CLASSES = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "del",
    "nothing",
    "space",
]

# 通用参数
INPUT_SHAPE = (3, 32, 32)
BATCH_SIZE = 128
EPOCHS = 15
TEST_SIZE = 0.2
VAL_SIZE = 0.1
RANDOM_SEED = 42
IMAGE_SIZE = (32, 32)
NUM_CLASSES = len(CLASSES)

# CNN特定参数
CNN_LEARNING_RATE = 0.0001  # 0.001

# 液态神经网络特定参数
LIQUID_LEARNING_RATE = 1e-4
CPU_THREADS = 16

# 数据变换参数
NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)

# 调试参数
TORCHINFO = False 
