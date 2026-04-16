# ECE452 Project 1, Part 1
# Group 8 Members: Arnav R., Raghav B., and Daniel A.
#
# Sample command: python3 Train.py --epochs 20 --lr 0.01 --plot plot.png --opt adam --loss mse 

# Inspiration comes from kaggle.com/code/geekysaint/solving-mnist-using-pytorch 
import re
import os
import sys
import torch
import argparse
import torchvision
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import random_split
from torchvision.transforms import ToTensor
from torch.utils.data import TensorDataset, DataLoader

#----------------------------
# Class for the neural network
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
            nn.Linear(64, 10),
            # I removed the softmax from here: CrossEntropyLoss applies it internally,
            # and raw logits work better with MSE too.
        )

    # Forward pass through convolutional and fully connected layers
    def forward(self, x):
        return self.Layers(x)

    # Training function. Uses Mean Squared Error loss with Stochastic
    # Gradient Descent optimizer.
    def fit(self, images, labels, epochs: int = 5, lr: float = 0.001, batch: int = 50, plot_name=None, loss_type='mse', opt_type='sgd'):
        images = torch.stack(images)
        labels = torch.tensor(labels)
        dataset = TensorDataset(images, labels)
        train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

        device = torch.device("cpu")
        self.to(device)

        if loss_type.lower() == 'mse':
            criterion = nn.MSELoss()
        else:
            criterion = nn.CrossEntropyLoss()
        
        if opt_type.lower() == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        else:
            optimizer = torch.optim.SGD(self.parameters(), lr=lr)

        # just puts the model in training mode, doesn't train it
        self.train()

        LossHistory = []
        for epoch in range(epochs):
            total_loss = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                labels_OneHot = F.one_hot(labels, num_classes=10).float()

                optimizer.zero_grad()
                outputs = self.forward(images)
                
                if loss_type.lower() == 'mse':
                    loss = criterion(outputs, labels_OneHot)
                else:
                    loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss/len(train_loader)
            LossHistory.append(avg_loss)
            print(f'Epoch [{epoch+1}/{epochs}] complete. Avg loss: {avg_loss:.4f}')

        if plot_name:
            self.PlotLoss(LossHistory, plot_name)

    def PlotLoss(self, LossHistory, plot_name):
        if not plot_name.endswith('.png'):
            plot_name += '.png'

        plt.figure(figsize=(10,5))
        plt.plot(LossHistory, color='red', label='Training Loss')
        plt.title(f'Training Performance for {plot_name}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig(plot_name)
        plt.close()
        print(f'Loss plot saved as {plot_name}')

#----------------------------
# Convert all training images to tensors 
#----------------------------
def ConvImageToTensor(RootFilePath: str) -> tuple[list[int], list[torch.FloatTensor]]:
    try:
        # Convert to Pillow image, and then use ToTensor function
        filenames = [f for f in os.listdir(RootFilePath) if f.endswith('.tif')]
        images = []
        labels = []
        for file in filenames:
            t = Image.open(os.path.join(RootFilePath, file))
            images.append(ToTensor()(t))
            labels.append(int(re.search(r'_(\d)_', file)[1]))

        return labels, images
    except FileNotFoundError:
        print(f'Specified root file path {RootFilePath} doesn\'t exist or smth')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Train MnistNetwork on MNIST training dataset")

    parser.add_argument('--dir', type=str, default='./Provided/Part1/training_data/', help='Path to TIF images.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
    parser.add_argument('--batch', type=int, default=64, help='Batch size for training')
    parser.add_argument('--output', type=str, default='model.pt', help='Name of output pt file')
    parser.add_argument('--plot', nargs='?', const='loss_plot.png', default=None, help='Enable plotting of loss curves.')
    parser.add_argument('--opt', choices=['sgd','adam'], default='sgd', help='Optimizer')
    parser.add_argument('--loss', choices=['mse','ce'], default='mse', help='Loss function')
    parser.add_argument('--neurons', type=int, default=128, help='Number of neurons in the first hidden layer')

    args = parser.parse_args()

    if args.neurons < 10:
        print('Number of neurons must be more than 10')
        sys.exit(1)

    labels, images = ConvImageToTensor(args.dir)

    model = MnistNetwork(args.neurons)
    model.fit(images, labels,
              epochs=args.epochs,
              lr=args.lr,
              batch=args.batch,
              plot_name=args.plot,
              loss_type=args.loss,
              opt_type=args.opt)

    finalname = args.output if args.output.endswith('.pt') else args.output + '.pt'
    torch.save(model, finalname)