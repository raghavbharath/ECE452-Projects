# ECE452 Project 1, Part 1
# Group 8 Members: Arnav R., Raghav B., and Daniel A.
#
# This test file implements a custom Neural Network, using NumPy for arrays.
# PyTorch is still imported, as it's the easiest way to extract weights from a .pt file

import re
import os
import sys
import torch
import argparse
import torchvision
import numpy as np
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import random_split
from torchvision.transforms import ToTensor
from sklearn.metrics import confusion_matrix
from torch.utils.data import TensorDataset, DataLoader

#----------------------------
# Class for a single layer
#----------------------------
class Layer():
    # Layers are stored in the shape of 1xN vectors.
    # Matrix multiplication will follow xA^T
    def __init__(self, length, layernum: int, activation: str = 'ReLU'):
        self.PreActivation = np.empty(length)
        self.PostActivation = np.empty(length)
        self.Bias = np.empty(length)
        self.ActFunc = activation
        self.LayerNum = layernum

    # Matrix multiplication, xA^T
    def MatMul(self, weights: np.ndarray, bias) -> np.ndarray:
        try:
            return np.dot(self.PostActivation, weights.T) + bias
        except Exception as e:
            print(f'Error while doing matrix multiplication of layer {self.LayerNum}: {e}')
            sys.exit(1)
    
    # Activate PreActivation to PostActivation
    def Activate(self):
        try:
            if self.ActFunc == 'ReLU':
                self.PostActivation = np.maximum(0, self.PreActivation)
            elif self.ActFunc == 'Sigmoid':
                self.PostActivation = 1 / (1 + np.exp(-self.PreActivation))
            elif self.ActFunc == 'Softmax':
                shift_z = self.PreActivation - np.max(self.PreActivation)
                exps = np.exp(shift_z)
                self.PostActivation = exps / np.sum(exps)
            elif self.ActFunc == 'None':
                self.PostActivation = self.PreActivation
            else:
                print(f'Invalid activation function {self.ActFunc}')
                print('Must be either \'ReLU\', \'Sigmoid\', \'Softmax\', or \'None\'')
                sys.exit(1)
        except Exception as e:
            print(f'Error while doing activation of layer {self.LayerNum}: {e}')
            sys.exit(1)

#----------------------------
# Class for the Network
#----------------------------
class Network():
    def __init__(self):
        self.layers = []
        self.weights = []

    # Layer numbering STARTS AT 0!!!
    def add_layer(self, length, activation='ReLU'):
        layernum = len(self.layers)
        new_layer = Layer(length, layernum, activation)
        self.layers.append(new_layer)

    def load_parameters(self, torch_model_path):
        # Use pytorch to load the weights into the network.
        # The pt files exported the entire model, not just the weights.

        # Load the full object
        try:
            full_model = torch.load(torch_model_path, weights_only=False)
            full_model.eval()

            linear_modules = [m for m in full_model.modules() if isinstance(m, torch.nn.Linear)]

            for i, module in enumerate(linear_modules):
                # Extract dimensions
                in_dim = module.in_features
                out_dim = module.out_features

                # Add the Input Layer if this is the very first weight matrix
                if i == 0:
                    self.add_layer(in_dim, activation='None') 

                # Add the subsequent layers
                act = 'Softmax' if i == len(linear_modules) - 1 else 'ReLU'
                self.add_layer(out_dim, activation=act)

                # Assign the parameters to the output Layer
                self.weights.append(module.weight.detach().numpy())
                self.layers[i+1].Bias = module.bias.detach().numpy()
        except Exception as e:
            print(f'Error loading network: {e}')
            sys.exit(1)

            print(f"Successfully initialized {len(self.layers)} layers from {torch_model_path}")

    def forward(self, input_vector: np.ndarray):
        """
        The core loop: 
        1. Set input of Layer 0
        2. Activate Layer 0
        3. MatMul Layer 0 -> result is PreActivation of Layer 1
        """
        try:
            # Step 1: Initialize the first layer's input
            # We flatten the input if it's an image (in our case 16x16 -> 256)
            self.layers[0].PreActivation = input_vector.flatten()
            
            # Step 2: Iterate through layers
            for i in range(len(self.layers) - 1):
                # Activate current layer (turns its Pre into Post)
                # Note: For the very first layer, Post is already set by the input
                self.layers[i].Activate()
                
                # If there's a next layer, perform MatMul to set the next layer's PreActivation
                if i + 1 < len(self.layers):
                    # We use the weights connecting Layer i to Layer i+1
                    next_pre = self.layers[i].MatMul(self.weights[i], self.layers[i+1].Bias)
                    self.layers[i+1].PreActivation = next_pre
            
            # return the final layer's PostActivation (the prediction)
            self.layers[-1].Activate()
            return self.layers[-1].PostActivation
        except Exception as e:
            print(f'Error on forward pass: {e}')

#----------------------------
# Class for original Pytorch network
# (it's giving issues otherwise)
#----------------------------
class MnistNetwork(nn.Module):
    def __init__(self, n: int = 128):
        super(MnistNetwork, self).__init__()
        self.Layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, n),
            nn.ReLU(),
            nn.Linear(n, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

#----------------------------
# Convert an image to a 16x16 np.ndarray with label
#----------------------------
def ConvImageToArray(RootFilePath: str) -> tuple[int, np.ndarray]:
    try:
        # Convert to Pillow image, and then convert to np array
        with Image.open(RootFilePath) as img:
            img_array = np.array(img, dtype=np.float32)
            img_array /= 255.0
            label = int(re.search(r'_(\d)_', RootFilePath)[1])
            return label, img_array
    except Exception as e:
        print(f'Error while converting image {RootFilePath} to array: {e}')
        sys.exit(1)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Test MnistNetwork on MNIST test dataset')

    parser.add_argument('--dir', type=str, default='./Provided/Part1/test_data', help='Path to test images folder')
    parser.add_argument('--conf', nargs='?', const='confusion_matrix.png', default=None, help='Enable plotting of confusion matrix')
    parser.add_argument('model', type=str, nargs='?', default='P1_Part1_ANN.pt', help='Path to pt file. Must contain structure of network.')

    args = parser.parse_args()

    try:
        net = Network()
        net.load_parameters(args.model)
    except Exception as e:
        print(f'Error loading model at {args.model}')
        sys.exit(1)

    print('Network initialized.')
    print(f'Beginning testing on {args.dir}')

    TotalImages = 0
    TotalCorrect = 0
    LabelsList = []
    PredictionList = []

    try:
        filenames = [os.path.join(args.dir, f) for f in os.listdir(args.dir) if f.endswith('.tif')]
        TotalImages = len(filenames)

        for file in filenames:
            label, img = ConvImageToArray(file)
            output = net.forward(img)
            prediction = np.argmax(output)
            print(f'File: {os.path.basename(file)} | Output layer: {np.round(output, 4)} | Predicted: {prediction} | True: {label}')
            LabelsList.append(label)
            PredictionList.append(int(prediction))
            if label == prediction:
                TotalCorrect += 1
    except Exception as e:
        print(f'Error occurred while running inference: {e}')
        sys.exit(1)

    print('Labels: ', LabelsList)
    print('Predictions: ', PredictionList)

    print(f'{TotalImages} tested, {TotalCorrect} correct, accuracy: {float(TotalCorrect)/TotalImages}')

    if args.conf:
        cm = confusion_matrix(LabelsList, PredictionList, labels=list(range(10)))

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=list(range(10)), 
                    yticklabels=list(range(10)))

        plt.title(f'MNIST Confusion Matrix, for {args.model}')
        plt.xlabel('Predicted Label')
        plt.ylabel('Actual Label')

        # Show/Save the plot
        plt.savefig(args.conf)
        plt.close()

