from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
from ikscode.IKSSW import IKSSW
from quantifiers.ApplyQtfs import ApplyQtfs
from timeit import default_timer as timer
import os

class DriftDetector(ABC):
    def __init__(self, stream, size_train, size_window, model, context_list):
      self.size_window = size_window
      self.context_list = context_list
      
      self.trainX = stream.iloc[:size_train, :-1]
      self.labels = stream.iloc[:size_train, -1]
      self.test = stream.iloc[size_train:, :-1]
      self.test.reset_index(inplace=True, drop=True)
      self.model = model.fit(self.trainX, self.labels)
  
      self.tw = pd.DataFrame()
      self.twlabels = []
      self.accs = []
      self.table = pd.DataFrame(columns=["IKS"])
      self.real_labels_window = 0
      self.vet_accs = {}

      self.table_accuracies = pd.DataFrame()
    
    @abstractmethod
    def runslidingwindow(self):
        pass

    def apply_qtf(self, new_instance):
      self.tw = pd.concat([self.tw, new_instance.to_frame().T], ignore_index=True)
      score = self.model.predict_proba(new_instance.to_frame().T)[:, 0]
      self.vet_accs[list(self.vet_accs.keys())[0]].append(self.model.predict(new_instance.to_frame().T)
                                                          .astype(int)[0])
      
      app = ApplyQtfs(self.trainX, self.labels.values.tolist(), self.tw, self.model, 0.5)
      proportions = app.aplly_qtf()
      
      for qtf, proportion in proportions.items():
        pos_scores = self.model.predict_proba(self.tw)[:, 0].tolist()
        thr = app.get_best_threshold(proportion, pos_scores)
        if qtf not in self.vet_accs:
          self.vet_accs[qtf] = self.twlabels.copy()
        self.vet_accs[qtf].append(1 if score >= thr else 0)

      self.vet_accs["real"] = self.real_labels_window
      print(pd.DataFrame(self.vet_accs))
      

    def get_real_proportion(self, index):
      self.real_labels_window = self.labels.iloc[index-len(self.tw): index+1]
      print(self.real_labels_window.value_counts(normalize=True))










      