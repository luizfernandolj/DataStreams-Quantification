from interface_class.DriftDetector import DriftDetector

import pandas as pd
import numpy as np
from ikscode.IKSSW import IKSSW


class Baseline(DriftDetector):
    
    def __init__(self, values, window_length):
        self.window_length = window_length
        
    def Increment(self, value, window, index):
        pass
    
    def Test(self, index):
        return False
    
    def Update(self, window):  
        pass