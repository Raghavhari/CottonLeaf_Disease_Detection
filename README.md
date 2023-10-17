# Cotton Leaf Disease Detection

This repository contains code for Cotton Leaf Disease Detection project using TensorFlow and Keras. The project leverages a pre-trained ResNet-152V2 model for image classification and includes data preprocessing, model training, and evaluation.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Getting Started](#getting-started)
- [Model Architecture](#model-architecture)
- [Training and Evaluation](#training-and-evaluation)
- [Customization](#customization)
- [Contribution](#contribution)

## Introduction

Image classification is a common computer vision task that involves assigning a label to an image based on its content. In this project, we use two popular pre-trained models, ResNet-152V2 and VGG16, for image classification. The code includes data preprocessing, model configuration, training, and evaluation.

## Requirements

Before running the code, ensure you have the following requirements:

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Scikit-learn

You can install the required packages using pip:

```bash
pip install tensorflow keras numpy matplotlib scikit-learn
```

## Getting Started

1. Clone the repository to your local machine:

```bash
git clone https://github.com/Raghavhari/Cotton_Leaf_Disease_Detection.git
```
2. Navigate to the project directory:

```bash
cd Cotton_Leaf_Disease_Detection
```
3. Set up your data directories and make sure to organize your training and test datasets accordingly.

4. Update the paths and parameters in the code to match your dataset and preferences.

5. Run the code to train and evaluate the model.

## Model Architecture
The project employs two renowned pre-trained models: ResNet-152V2 and VGG16. These models serve as the foundation for image classification and can be further customized based on the problem at hand.

## Training and Evaluation

To train the model, execute the Python script:

```bash
Transfer Learning Resnet152V2.ipynb
```
You can adjust hyperparameters, including the number of epochs, batch size, and learning rate, within the code. After training, the code provides comprehensive evaluation metrics, including loss, accuracy.

## Customization
Feel free to customize the code to fit your specific image classification project. You can add custom layers, experiment with different pre-trained models, and incorporate data augmentation techniques to enhance model performance.

## Contribution
If you wish to contribute to the project or report issues, please open a GitHub issue or submit a pull request. We appreciate your input and collaboration.
