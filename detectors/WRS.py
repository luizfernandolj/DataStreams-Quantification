from interface_class.DriftDetector import DriftDetector

import pandas as pd
import numpy as np
from scipy import stats

class WRS(DriftDetector):

  def __init__(self, stream, size_train, size_window, model, context_list, threshold):
    super().__init__(stream, size_train, size_window, model, context_list)
    self.threshold = threshold
    self.drift_points = []
    self.flag = False

  def runslidingwindow(self):

    drift_points = []
    self.vet_accs["WRS"] = []
    
    w1 = self.trainX.iloc[-self.size_window:].copy()
    
    for index, new_instance in self.test.iterrows():
      print('Example {}/{} drifts: {}'.format(index+len(self.trainX)+1, 
                                              len(self.labels)+len(self.test),
                                              self.drift_points), end='\r')
      print("")
      if len(self.tw) >= 10:
        self.get_real_proportion(index)
        self.apply_qtf(new_instance)

        if len(self.tw) == self.size_window:
          w2 = self.tw.copy()
          w2_labels = self.real_labels_window.copy()
          _, n_features = self.tw.shape
            
          accuracies = self.compute_accuracies()
          print("================================")
          self.detect_drift(index, n_features, w1, w2, w2_labels)

      else:
        self.vet_accs["WRS"].append(self.model.predict(new_instance.to_frame().T).astype(int)[0])
        self.twlabels = self.vet_accs["WRS"].copy()
        self.tw = pd.concat([self.tw, new_instance.to_frame().T], ignore_index=True)
    vet_accs = pd.concat([pd.DataFrame(self.final_vet_accs), self.test], axis=1, ignore_index=True)
    return accuracies, vet_accs 


  def detect_drift(self, index, n_features, w1, w2, w2_labels):
        
    for j in range(0, n_features):
      _, p_value = stats.ranksums(w1.iloc[:,j], w2.iloc[:,j])        
      if (p_value <= self.threshold):
          self.flag = True

    if self.flag:
      self.drift_points.append(index+len(self.trainX))
      w1 = w2 # update the reference window with recent data of w2
      self.flag = False
      self.trainX = self.tw.copy()
      self.trainy = self.real_labels_window.copy()
      self.tw = pd.DataFrame()
      self.vet_accs = {"WRS":[]}
      self.model.fit(self.trainX, self.trainy)  
    





