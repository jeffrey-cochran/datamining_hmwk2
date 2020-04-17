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
    torchvision.datasets.MNIST(
        "/data/test", 
        download=False, 
        train=True, 
        transform=my_transform,
        target_transform=OneHOT()
    )
)
#
# Load test data
MNIST_Data_test = DataStreamer(
    torchvision.datasets.MNIST(
        "/data/test", 
        download=False, 
        train=False, 
        transform=my_transform,
        target_transform=OneHOT()
    )
)
#
# Set tolerance and step size
# terminate when loss improvement is
# relatively small
rel_tol = 1e-3
batch_size = 100
max_iter=1000
#
# Iterate over step sizes
step_sizes = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
for step_size in step_sizes:
    #
    # Set batch sizes
    MNIST_Data.set_batch_size(batch_size)
    #
    # Init function definitions
    # and randomize model parameters
    F = ForwardModel()
    L = LossFunction(F)
    #
    print(f'Iterating with step size {step_size}...')
    #
    # Create CSV file for data collection
    fname = f"step_{step_size:1.0e}_batch_{batch_size}.csv"
    with open(fname, "w", newline="") as f:
        csv_file = csv.writer(f, delimiter=',')
        #
        # Iterate to convergence
        N = float(len(MNIST_Data))
        eval_counter = 0
        train_loss = L(MNIST_Data.X, MNIST_Data.Y)
        train_loss_delta = train_loss
        while eval_counter <= max_iter:
            #
            # Don't bother recording
            # every 
            if eval_counter % 10 == 0:
                #
                # Update convergence criterion check
                prev_train_loss = train_loss
                train_loss = L(MNIST_Data.X, MNIST_Data.Y)
                # print(f"Train loss: {train_loss}")
                train_loss_delta = abs(train_loss - prev_train_loss)
                test_loss = L(MNIST_Data_test.X, MNIST_Data_test.Y)
                #
                # Write progress
                csv_file.writerow([train_loss.item(), test_loss.item(), eval_counter])
                #
            #
            # Iterate over batches
            j = 0
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
            #
        #
    #