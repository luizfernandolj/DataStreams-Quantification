from interface_class.DriftDetector import DriftDetector

import pandas as pd
import numpy as np
from scipy import stats

class WRS(DriftDetector):
    
    def __init__(self, values, window_length, threshold):
        self.window_length = window_length
        self.threshold = threshold
        self.w1 = values.iloc[-window_length:, :-1].copy()
        self.w2 = values.iloc[-window_length:, :-1].copy()
        self.w2_labels = values.iloc[-window_length:].copy()
        self.r, self.n_features = values.iloc[:, :-1].shape
        
    def Increment(self, value, window, index):
        self.w2.drop(self.w2.index[0], inplace=True, axis=0)
        self.w2 = pd.concat([self.w2, value], ignore_index=True)
        self.w2_labels.drop(self.w2_labels.index[0], inplace=True, axis=0)
        self.w2_labels = pd.concat([self.w2_labels, value], ignore_index=True)
    
    def Test(self, index):
        for j in range(0, self.n_features):
             _, p_value = stats.ranksums(self.w1.iloc[:,j], self.w2.iloc[:,j])        
        if (p_value <= self.threshold):
            return True
    
    def Update(self, window):  
        self.w1 = self.w2