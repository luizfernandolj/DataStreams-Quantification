from interface_class.DriftDetector import DriftDetector

import pandas as pd
import numpy as np
from ikscode.IKSSW import IKSSW
from timeit import default_timer as timer
import os

class IKS(DriftDetector):

  def __init__(self, stream, size_train, size_window, model, context_list, ca):
    super().__init__(stream, size_train, size_window, model, context_list)
    self.ca = ca

  def runslidingwindow(self):
    print(self.tw)

  def apply_qtf(self, new_instance):
    pass