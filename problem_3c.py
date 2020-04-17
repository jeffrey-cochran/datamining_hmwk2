from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision
from transforms import ReshapeImage, OneHOT
from DataStreamer_def import DataStreamer
from ForwardModel_def import ForwardModel
from LossFunction_def import LossFunction
from CrossValidationStreamer_def import CrossValidationStreamer
from math import sqrt
from torch import cuda, __version__
import csv
#
# Define transform of image data
my_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    ReshapeImage()
])
#
# Load training data
MNIST_Data = DataStreamer(
    data=torchvision.datasets.MNIST(
        "/data/test", 
        download=True, 
        train=True, 
        transform=my_transform,
        target_transform=OneHOT()
    )
)
#
# Load test data
MNIST_Data_test = DataStreamer(
    data=torchvision.datasets.MNIST(
        "/data/test", 
        download=True, 
        train=False, 
        transform=my_transform,
        target_transform=OneHOT()
    )
)
#
# Set tolerance and step size
# terminate when loss improvement is
# relatively small
max_iter= 500
#
# Set params
batch_size=60000
ridge_coeff=0.01
step_size=0.001
#
# Set batch sizes
MNIST_Data.set_batch_size(batch_size)
#
# Init function definitions
# and randomize model parameters
F = ForwardModel()
L = LossFunction(F, ridge_coeff=ridge_coeff)
#
# Iterate to convergence
eval_counter = 0
while eval_counter <= max_iter:
    #
    # Create CSV file for data collection
    fname = f"batch_size_{batch_size:1.0e}_step_size_{step_size:1.0e}_ridge_coeff_{ridge_coeff:1.0e}.csv"
    with open(fname, "a", newline="") as f:
        
        train_loss = L(MNIST_Data.X, MNIST_Data.Y)
        test_loss = L(MNIST_Data_test.X, MNIST_Data_test.Y)
        #
        # Write progress
        csv_file = csv.writer(f, delimiter=',')
        csv_file.writerow([
            train_loss.item(), 
            test_loss.item(), 
            eval_counter
        ])
        #
    #
    # Iterate over batches
    for (x, y) in MNIST_Data:
        #
        # Compute gradient and gradient norm
        grad_W, grad_w0 = L.gradient(x, y)
        cuda.synchronize()
        #
        # Update weights
        cuda.synchronize()
        F.update(grad_W, grad_w0, step_size)
    #
    eval_counter += 1