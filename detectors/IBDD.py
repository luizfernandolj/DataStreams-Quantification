from interface_class.DriftDetector import DriftDetector

import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib as mpl
from skimage.io import imread
import math
from skimage.metrics import mean_squared_error, structural_similarity
from random import seed, shuffle


class IBDD(DriftDetector):
    
    def __init__(self, values, consecutive_values, window_length, dataset):
        self.fname = f"{os.getcwd()}/ibdd_folder/{dataset}"
        self.reference = values
        self.window_length = window_length
        self.n_runs = 20
        self.consecutive_values = consecutive_values
        self.superior_threshold, self.inferior_threshold, self.nrmse = self.find_initial_threshold(self.reference, window_length, self.n_runs)
        self.threshold_diffs = [self.superior_threshold - self.inferior_threshold]
        self.w1 = self.get_imgdistribution(f"{self.fname}/w1.jpeg", self.reference.iloc[-window_length:])
        self.last_update = 0
        
    def find_initial_threshold(self, X_train, window_length, n_runs):
        if window_length > len(X_train):
            window_length = len(X_train)

        w1 = X_train.iloc[-window_length:].copy()
        w1_cv = self.get_imgdistribution(f"{self.fname}/w1_cv.jpeg", w1)

        max_index = X_train.shape[0]
        sequence = [i for i in range(max_index)]
        nrmse_cv = []
        for i in range(0,n_runs):
            # seed random number generator
            seed(i)
            # randomly shuffle the sequence
            shuffle(sequence)
            w2 = X_train.iloc[sequence[:window_length]].copy()
            w2.reset_index(drop=True, inplace=True)
            w2_cv = self.get_imgdistribution(f"{self.fname}/w2_cv.jpeg", w2)
            nrmse_cv.append(mean_squared_error(w1_cv,w2_cv))
            threshold1 = np.mean(nrmse_cv)+2*np.std(nrmse_cv)
            threshold2 = np.mean(nrmse_cv)-2*np.std(nrmse_cv)
        if threshold2 < 0:
            threshold2 = 0		
        return (threshold1, threshold2, nrmse_cv)   
    
    
    def get_imgdistribution(self, name_file, data):
        plt.imsave(name_file, data.transpose(), cmap = 'Greys', dpi=100)
        w = imread(name_file)
        return w
    
    def Increment(self, value, window, index):
        w2 = self.get_imgdistribution(f"{self.fname}/w2.jpeg", window)

        self.nrmse.append(mean_squared_error(self.w1, w2))
        
        if (index - self.last_update > 60):
            self.superior_threshold = np.mean(self.nrmse[-50:]) + 2 * np.std(self.nrmse[-50:])
            self.inferior_threshold = np.mean(self.nrmse[-50:]) - 2 * np.std(self.nrmse[-50:])
            self.threshold_diffs.append(self.superior_threshold - self.inferior_threshold)
            self.last_update = index
    
    def Test(self, index):
        if (all(i >= self.superior_threshold for i in self.nrmse[-self.consecutive_values:])):
            self.last_update = index
            return True

        elif (all(i <= self.inferior_threshold for i in self.nrmse[-self.consecutive_values:])):
            self.last_update = index
            return True
        return False
    
    
    def Update(self, window):  
        if (all(i >= self.superior_threshold for i in self.nrmse[-self.consecutive_values:])):
            self.superior_threshold = self.nrmse[-1] + np.std(self.nrmse[-50:-1])
            self.inferior_threshold = self.nrmse[-1] - np.mean(self.threshold_diffs)
            self.threshold_diffs.append(self.superior_threshold - self.inferior_threshold)

        elif (all(i <= self.inferior_threshold for i in self.nrmse[-self.consecutive_values:])):
            self.inferior_threshold = self.nrmse[-1] - np.std(self.nrmse[-50:-1])
            self.superior_threshold = self.nrmse[-1] + np.mean(self.threshold_diffs)
            self.threshold_diffs.append(self.superior_threshold - self.inferior_threshold)
        self.w1 = self.get_imgdistribution(f"{self.fname}/w1_cv.jpeg", window)
    
    