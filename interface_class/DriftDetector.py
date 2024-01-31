from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
from ikscode.IKSSW import IKSSW
from timeit import default_timer as timer
import os

class DriftDetector(ABC):
    def __init__(self, stream, size_train, size_window, model, context_list):
      self.size_window = size_window
      self.model = model
      self.context_list = context_list
      
      self.trainX = stream.iloc[:size_train, :-1]
      self.labels = stream.iloc[:size_train, -1]
      self.test = stream.iloc[size_train:, :-1] 
  
      self.tw = []
      self.accs = []
      self.table = pd.DataFrame(columns=["IKS"])
    
    @abstractmethod
    def runslidingwindow(self):
        pass

    @abstractmethod
    def apply_qtf(self, new_instance):
        pass