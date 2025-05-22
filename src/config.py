"""统一配置参数"""

# 数据路径
TRAIN_DIR = "dataset/asl_alphabet_train/asl_alphabet_train"
TEST_DIR = "dataset/asl_alphabet_test/asl_alphabet_test"
MODEL_SAVE_DIR = "output/"

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

# 优化器和训练参数
OPTIMIZER_TYPE = "adam"  # 'adam', 'sgd', 'adamw'
WEIGHT_DECAY = 1e-5
USE_SCHEDULER = False
SCHEDULER_TYPE = "cosine"  # 'cosine', 'plateau'
EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 5

# CNN特定参数
CNN_LEARNING_RATE = 1e-4
CNN_DROPOUT = 0.2

# 液态神经网络特定参数
LIQUID_LEARNING_RATE = 1e-4
LIQUID_SOLVER = "dopri5"  # 'euler', 'midpoint', 'rk4', 'dopri5'
CPU_THREADS = 16

# ViT特定参数
VIT_LEARNING_RATE = 5e-5
VIT_PATCH_SIZE = 8
VIT_EMBED_DIM = 384
VIT_DEPTH = 8
VIT_NUM_HEADS = 6
VIT_MLP_RATIO = 4.0
VIT_DROPOUT = 0.1

# 数据变换参数
NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)

# 调试参数
TORCHINFO = True
