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
from torch import cuda
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
step_sizes = 10**(np.linspace(-2, -3, 5))
max_iter=5000
#
# Iterate over step_sizes
for step_size in step_sizes:
    #
    # Init function definitions
    # and randomize model parameters
    F = ForwardModel()
    L = LossFunction(F)
    #
    print(f'Iterating with step size {step_size}...')
    #
    # Create CSV file for data collection
    fname = f"step_{step_size:1.0e}_tol_{rel_tol:1.0e}.csv"
    with open(fname, "w", newline="") as f:
        csv_file = csv.writer(f, delimiter=',')
        #
        # Iterate to convergence
        N = float(len(MNIST_Data))
        eval_counter = 0
        tolerance = 0
        train_loss = L(MNIST_Data.X, MNIST_Data.Y)
        train_loss_delta = train_loss
        while eval_counter <= max_iter:
            #
            # Compute gradient and gradient norm
            grad_W, grad_w0 = L.gradient(MNIST_Data.X, MNIST_Data.Y)
            cuda.synchronize()
            #
            # Update weights
            cuda.synchronize()
            F.update(grad_W, grad_w0, step_size)
            if eval_counter % 10 == 0:
                #
                # Update convergence criterion check
                prev_train_loss = train_loss
                train_loss = L(MNIST_Data.X, MNIST_Data.Y)
                train_loss_delta = abs(train_loss - prev_train_loss)
                test_loss = L(MNIST_Data_test.X, MNIST_Data_test.Y)
                #
                # Write progress
                csv_file.writerow([train_loss.item(), test_loss.item(), eval_counter])
                # print(f'Training Loss: {train_loss} \nTesting Loss: {test_loss} \nIters: {eval_counter} \n===========')
                
            eval_counter += 1
            # print(eval_counter)

        # z = torch.eye(3)
        # zz = torch.zeros((3,3))
        # zz[0][0] = 1

        # zzz = z - zz
        # nz = zzz.max(0).values.nonzero()
        # print(zzz)
        # print(len(nz))
        # print(torch.sum(zzz, dim=0))