from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from quantifiers.ApplyQtfs import ApplyQtfs
from timeit import default_timer as timer
from sklearn.metrics import accuracy_score
import os

class DriftDetector(ABC):
    def __init__(self, stream, size_train, size_window, model, context_list):
      self.size_window = size_window
      self.context_list = context_list
      
      self.trainX = stream.iloc[:size_train, :-1].copy()
      self.labels = stream.iloc[:size_train, -1].copy()
      self.trainy = self.labels.copy()
      self.test = stream.iloc[size_train:, :-1].copy()
      self.labels_test = stream.iloc[size_train:, -1].copy()
      self.labels_test.reset_index(inplace=True, drop=True)
      self.test.reset_index(inplace=True, drop=True)
      self.model = model.fit(self.trainX, self.labels)
  
      self.tw = pd.DataFrame()
      self.twlabels = []
      self.accs = []
      self.table = pd.DataFrame()
      self.real_labels_window = 0
      self.vet_accs = {}
    
    @abstractmethod
    def runslidingwindow(self):
        pass

    def apply_qtf(self, new_instance):
      if len(self.tw) < self.size_window:
        self.tw = pd.concat([self.tw, new_instance.to_frame().T], ignore_index=True)
      else:
        self.tw = pd.concat([self.tw, new_instance.to_frame().T], ignore_index=True)[1:]
        self.vet_accs[list(self.vet_accs.keys())[0]].pop(0)  
      
      score = self.model.predict_proba(new_instance.to_frame().T)[:, 1]

      self.vet_accs[list(self.vet_accs.keys())[0]].append(self.model.predict(new_instance.to_frame().T)
                                                          .astype(int)[0])
      
      app = ApplyQtfs(self.trainX, self.trainy, self.tw, self.model, 0.5)
      proportions = app.aplly_qtf()
      print(proportions)
      
      for qtf, proportion in proportions.items():
        name = f"{list(self.vet_accs.keys())[0]}-{qtf}"
        pos_scores = self.model.predict_proba(self.tw)[:,1].tolist()
        thr = app.calc_threshold(proportion, pos_scores)
        print(f"proportion:{proportion} /// thr:{thr}, score:{score}")
        if name not in self.vet_accs:
          self.vet_accs[name] = self.twlabels.copy()
        self.vet_accs[name].append(1 if score >= thr else 0)
        if len(self.vet_accs[name]) == self.size_window+1:   
          self.vet_accs[name].pop(0)
  
      self.vet_accs["real"] = self.real_labels_window.tolist()
      print(pd.DataFrame(self.vet_accs))
      

    def get_real_proportion(self, index):
      start =  index-len(self.tw)+1 if len(self.tw) >= self.size_window else index-len(self.tw)
      self.real_labels_window = self.labels_test.iloc[start: index+1]
      print(self.real_labels_window.value_counts(normalize=True).tolist())

    def compute_accuracies(self):
      d = {}
      for key, vet_acc in self.vet_accs.items():
        d[key] = round(accuracy_score(self.real_labels_window, vet_acc), 2)
      self.table = pd.concat([self.table, pd.DataFrame([d])])
      self.table.drop(columns=["real"], inplace=True)
      self.table.reset_index(inplace=True, drop=True)
      return self.table.round(2)
    
    def plot_acc(self):
      fig, ax = plt.subplots(figsize=(4, 2))
      for key, accs  in self.vet_accs.items():
        plt.plot([float(x)*self.size_window for x in range(0,len(accs))], accs, label=key )
      ax.spines['top'].set_visible(False)
      ax.spines['right'].set_visible(False)
      plt.xlabel('Examples')
      plt.ylabel('Accuracy') 
      plt.legend()









      