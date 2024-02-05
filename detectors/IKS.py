from interface_class.DriftDetector import DriftDetector

import pandas as pd
import numpy as np
from ikscode.IKSSW import IKSSW

class IKS(DriftDetector):

  def __init__(self, stream, size_train, size_window, model, context_list, ca):
    super().__init__(stream, size_train, size_window, model, context_list)
    self.ca = ca
    self.ikssw = IKSSW(self.trainX.values.tolist()[-size_window:])
    self.drift_points = []

  def runslidingwindow(self):

    drift_points = []
    self.vet_accs["IKS"] = []
    
    for index, new_instance in self.test.iterrows():
      print('Example {}/{} drifts: {}'.format(index+len(self.trainX)+1, 
                                              len(self.labels)+len(self.test),
                                              self.drift_points), end='\r')
      print("")
      self.ikssw.Increment(new_instance.values.tolist())
      if len(self.tw) >= 10:
        self.get_real_proportion(index)
        self.apply_qtf(new_instance)
        if len(self.tw) < self.size_window:
          self.tw = pd.concat([self.tw, new_instance.to_frame().T], ignore_index=True)

        if len(self.tw) == self.size_window:
          self.tw = pd.concat([self.tw, new_instance.to_frame().T], ignore_index=True)[1:]
          print(self.compute_accuracies())
          print("================================")
          self.detect_drift(self.ca, index)

      else:
        self.vet_accs["IKS"].append(self.model.predict(new_instance.to_frame().T).astype(int)[0])
        self.twlabels = self.vet_accs["IKS"].copy()
        self.tw = pd.concat([self.tw, new_instance.to_frame().T], ignore_index=True)


  def detect_drift(self, index, ca = 1.95):
    is_drift = self.ikssw.Test(ca)
    if is_drift:
      print("is drift")

      self.drift_points.append(index+len(self.trainX))
      
      self.ikssw.Update()
      self.trainX = self.tw.copy()
      self.trainy = self.real_labels_window.copy()
      self.tw = pd.DataFrame()
      self.vet_accs = {"IKS":[]}
      self.model.fit(self.trainX, self.trainy)
    





