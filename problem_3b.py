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
        download=False, 
        train=True, 
        transform=my_transform,
        target_transform=OneHOT()
    )
)
#
# Create cross validation folds
num_folds = 5
folds = CrossValidationStreamer(MNIST_Data, num_folds)
#
# Load test data
# MNIST_Data_test = DataStreamer(
#     data=torchvision.datasets.MNIST(
#         "/data/test", 
#         download=False, 
#         train=False, 
#         transform=my_transform,
#         target_transform=OneHOT()
#     )
# )
#
# Set tolerance and step size
# terminate when loss improvement is
# relatively small
max_iter= 500
#
# Grid search with cross validation
ridge_coeffs = [0.01]
step_sizes = [1e-3]
batch_sizes = [30, 100]
for ridge_coeff in ridge_coeffs:
    for step_size in step_sizes:
        for batch_size in batch_sizes:
            print(f'Iterating with...\n\t...batch_size[{batch_size}],\n\t...step_size[{step_size}],\n\t...ridge_coeff[{ridge_coeff}]')
            #
            # Evaluate average error over folds
            avg_error = 0
            fold_number = 0
            for train, test in folds:
                #
                fold_number += 1
                print(f"Evaluating fold #{fold_number}")
                #
                # Set batch sizes
                train.set_batch_size(batch_size)
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
                    # Iterate over batches
                    for (x, y) in train:
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
                #  Update average error
                avg_error += L(test.X, test.Y)
                #
            #
            # 
            avg_error = avg_error / float(num_folds)
            # Create CSV file for data collection
            fname = f"cross_val.csv"
            with open(fname, "a", newline="") as f:
                #
                # Write progress
                csv_file = csv.writer(f, delimiter=',')
                csv_file.writerow([
                    ridge_coeff, 
                    batch_size, 
                    step_size,
                    avg_error
                ])
                #