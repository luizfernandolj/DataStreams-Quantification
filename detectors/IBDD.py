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

font = {'weight': 'normal', 'size': 13}
mpl.rc('font', **font)

mpl.rcParams['figure.figsize'] = (6, 4)  # (6.0,4.0)
mpl.rcParams['font.size'] = 12  # 10
mpl.rcParams['savefig.dpi'] = 100  # 72
mpl.rcParams['figure.subplot.bottom'] = .11  # .125

class IBDD(DriftDetector):

  def __init__(self, stream, size_train, size_window, model, context_list, consecutive_values):
    super().__init__(stream, size_train, size_window, model, context_list)
    self.drift_points = []
    self.n_runs = 20
    self.consecutive_values = consecutive_values
    self.files2del = ['w1.jpeg', 'w2.jpeg', 'w1_cv.jpeg', 'w2_cv.jpeg']

  def runslidingwindow(self):

    drift_points = []
    self.vet_accs["IKS"] = []
    superior_threshold, inferior_threshold, nrmse = self.find_initial_threshold(self.trainX, self.size_window, self.n_runs)
    threshold_diffs = [superior_threshold - inferior_threshold]
    
    w1 = self.get_imgdistribution("w1.jpeg", self.trainX.iloc[-self.size_window:].copy())
    lastupdate = 0
    for index, new_instance in self.test.iterrows():
      print('Example {}/{} drifts: {}'.format(index+len(self.trainX)+1, 
                                              len(self.labels)+len(self.test),
                                              self.drift_points), end='\r')
      print("")
      if len(self.tw) >= 10:
        self.get_real_proportion(index)
        self.apply_qtf(new_instance)

        if len(self.tw) == self.size_window:
          w2 = self.get_imgdistribution("w2.jpeg", self.tw)
          nrmse.append(mean_squared_error(w1, w2))
          accuracies = self.compute_accuracies()
          print("================================")
          lastupdate, inferior_threshold, superior_threshold, threshold_diffs, nrmse = self.detect_drift(index, lastupdate, superior_threshold, inferior_threshold, threshold_diffs, nrmse)

      else:
        self.vet_accs["IKS"].append(self.model.predict(new_instance.to_frame().T).astype(int)[0])
        self.twlabels = self.vet_accs["IKS"].copy()
        self.tw = pd.concat([self.tw, new_instance.to_frame().T], ignore_index=True)
    self.plot_acc()
    for f in self.files2del:
      os.remove(f)
    return accuracies


  def detect_drift(self, index, lastupdate, superior_threshold, inferior_threshold, threshold_diffs, nrmse):
    if (index - lastupdate > 60):
      superior_threshold = np.mean(nrmse[-50:]) + 2 * np.std(nrmse[-50:])
      inferior_threshold = np.mean(nrmse[-50:]) - 2 * np.std(nrmse[-50:])
      threshold_diffs.append(superior_threshold - inferior_threshold)
      lastupdate += 1
        
    if (all(i >= superior_threshold for i in nrmse[-self.consecutive_values:])):
      superior_threshold = nrmse[-1] + np.std(nrmse[-50:-1])
      inferior_threshold = nrmse[-1] - np.mean(threshold_diffs)
      threshold_diffs.append(superior_threshold - inferior_threshold)
      self.drift_points.append(index+len(self.trainX))
      self.trainX = self.tw.copy()
      self.trainy = self.real_labels_window.copy()
      self.tw = pd.DataFrame()
      self.vet_accs = {"IKS":[]}
      self.model.fit(self.trainX, self.trainy)
      lastupdate += 1

    elif (all(i <= inferior_threshold for i in nrmse[-self.consecutive_values:])):
      inferior_threshold = nrmse[-1] - np.std(nrmse[-50:-1])
      superior_threshold = nrmse[-1] + np.mean(threshold_diffs)
      threshold_diffs.append(superior_threshold - inferior_threshold)
      self.drift_points.append(index+len(self.trainX))
      self.trainX = self.tw.copy()
      self.trainy = self.real_labels_window.copy()
      self.tw = pd.DataFrame()
      self.vet_accs = {"IKS":[]}
      self.model.fit(self.trainX, self.trainy)
      lastupdate += 1
    return lastupdate, inferior_threshold, superior_threshold, threshold_diffs, nrmse


  def find_initial_threshold(self, X_train, window_length, n_runs):
    
      w1 = X_train.iloc[-window_length:].copy()
      w1_cv = self.get_imgdistribution("w1_cv.jpeg", w1)

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
          w2_cv = self.get_imgdistribution("w2_cv.jpeg", w2)
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





