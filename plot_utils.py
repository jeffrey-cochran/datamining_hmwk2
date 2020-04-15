import csv
import pandas as pd
import matplotlib.pyplot as plt
from constants import TRAIN, TEST

class Series(object):

    def __init__(self, x=None, y=None, label=None):
        
        self.x = []
        if x is not None:
            self.x = x

        self.y = []
        if y is not None:
            self.y = y
        
        self.label = ""
        if label is not None:
            self.label = label
        
        return

    def load_from_file(self, file_name, label, column):
        #
        # Load data
        with open(file_name, newline="") as f:
            self.x = []
            self.y = []
            self.label = label
            csv_reader = csv.reader(f, delimiter=",")
            for row in csv_reader:
                self.y.append(float(row[column]))
                self.x.append(float(row[2]))

        return

    def plot(self, ax):
        ax.plot(self.x, self.y,label=self.label)
        return
