# DynamicHandSign: ASL Recognition with Neural ODEs

This project implements American Sign Language (ASL) recognition using deep learning and PyTorch. It provides two neural network architectures:

- Standard Convolutional Neural Network (CNN)
- Liquid Neural Network based on Neural Ordinary Differential Equations (ODEs)

## Features
- ASL recognition from images
- Two model architectures: CNN and Liquid Neural Network
- Model performance visualization tools
- Both models can achieve an accuracy of over 99.8% on the ASL dataset with appropriate training.


## Dataset Preparation
This project uses an American Sign Language (ASL) letter image dataset. Each image is a hand sign corresponding to a specific letter.

1. Download the dataset: [ASL Alphabet Dataset on Kaggle ](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
2. Create a folder named `dataset/` in the root directory
3. Place data folders `asl_alphabet_train/` and `asl_alphabet_test/` inside the dataset folder you just created

## Dependencies

`matplotlib`>=3.7.0
`opencv_python`>=4.5.0
`scikit_learn`>=1.0.0
`torch`>=2.0.0
`torchdyn`>=0.9.0
`torchvision`>=0.15.0

## Project Structure
```
ASL-Recognition
├─ dataset/
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
## Model Architectures

#### CNN Model
A standard Convolutional Neural Network with the following architecture:

- 3 convolutional layers (64, 128, 256 filters)
- Max-pooling and batch normalization after each convolutional layer
- Dropout layers for regularization
- Fully connected layers for classification

#### Liquid Neural Network
An advanced architecture based on Neural ODEs:

- CNN feature extraction front-end (similar to standard CNN)
- Dynamic layer based on Neural ODE
- Linear classification head


## Usage
```Python
# Train using the standard CNN model (default)
python main.py

# Train using the Liquid Neural Network
python main.py --model liquid

# Force using CPU even if GPU is available
python main.py --cpu-only

# Visualize sample images before training
python main.py --visualize-samples
```

## Command Line Arguments

| Argument              | Description                                   | Default |
| --------------------- | --------------------------------------------- | ------- |
| `--model`             | Choose model architecture (`cnn` or `liquid`) | `cnn`   |
| `--cpu-only`          | Force using CPU even if GPU is available      | None    |
| `--visualize-samples` | Visualize sample images before training       | None    |


## Training Process
- Load and preprocess data (resize to 32x32 pixels and normalize)
- Initialize model based on specified architecture
- Train using appropriate optimizer and loss function
- Evaluate model performance on validation set (for standard CNN)
- Visualize training metrics and save model
- Evaluate model on test set

## Model Outputs
After training, models are saved as:

- CNN model: `output/asl_cnn_model.pth`
- Liquid Neural Network: `output/asl_liquid_model.pth`

## Performance Visualization
The training script automatically generates visualization charts for:
- Training and validation loss curves
- Training and validation accuracy curves

