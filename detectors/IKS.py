from interface_class.DriftDetector import DriftDetector

import pandas as pd
import numpy as np
from ikscode.IKSSW import IKSSW


class IKS(DriftDetector):
    
    def __init__(self, values, window_length, ca):
        self.window_length = window_length
        self.ca : float = ca
        self.ikssw = IKSSW(values.iloc[-window_length:, :-1].values.tolist())
        
    def Increment(self, value, window, index):
        self.ikssw.Increment(value.values.tolist())
    
    def Test(self, index):
        return self.ikssw.Test(self.ca)
    
    def Update(self, window):  
        self.ikssw.Update()
    
    