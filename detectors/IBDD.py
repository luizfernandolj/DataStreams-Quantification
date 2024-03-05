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
  """
  The IBDD () class is a drift detector to detect if the data has changed over time, inheriting from DriftDetector class
  
  ...
  
  Attributes
  ----------
  train : pandas.core.frame.DataFrame
      pandas Dataframe containing the train data to be passed as argument to the model
  test : pandas.core.frame.Dataframe
      pandas Dataframe containing the test data to be passed as test to the model predict
  size_window : int
      the size of the sliding window
  model : object
      a soft classifier to predict scores of the sliding window
  consecutive_values : int
      #TODO
      
  Methods
  -------
  run_sliding_window():
      Simulates the instances coming one at a time
  detect_drift(ca : float) -> bool
      abstract method that makes the Kolmogorov-Smirnov test to detect if a drift has occured
  """
  

  def __init__(self, train, test, size_window, model, consecutive_values, ibdd_dir):
    """
    Construct all the necessacy attributes for the class IKS

    Parameters
    ----------
        train : pandas.core.frame.DataFrame
            pandas Dataframe containing the train data to be passed as argument to the model
        test : pandas.core.frame.Dataframe
            pandas Dataframe containing the test data to be passed as test to the model predict
        size_window : int
            the size of the sliding window
        model : object
            a soft classifier to predict scores of the sliding window
        consecutive_values : int
            #TODO
        n_runs : int
            #TODO
        lies2del : list[str]
            name of the files to delete, because they are only needed one time
    """
    super().__init__(train, test, size_window, model)
    self.lastupdate = 0
    self.inferior_threshold = 0
    self.superior_threshold = 0
    self.threshold_diffs = []
    self.nrmse = []
    self.n_runs = 20
    self.consecutive_values = consecutive_values
    self.ibdd_dir = ibdd_dir
    self.files2del = ['w1.jpeg', 'w2.jpeg', 'w1_cv.jpeg', 'w2_cv.jpeg']

  def run_sliding_window(self) -> list:
    """
    Method to simulate instances coming one at a time, from a DataStream, applying the Kolmogorov-Smirnov Test and applying quantification
    
    Returns
    -------
    vet_accs : dict[str : list[int]]
        predictions of the model along all datastream, with the quantification predictions
    drift_points : list[int]
        positions where the drift detector detected a drift
    tw_proportions = list[float]
        proportions of all windows 
    """
    drift_points : list[int] = []
    vet_accs : dict[str:list[int]] = {"IBDD": []}
    
    
    self.superior_threshold, self.inferior_threshold, self.nrmse = self.find_initial_threshold(self.train.iloc[:, :-1].copy(), 
                                                                                               self.size_window, 
                                                                                               self.n_runs)
    threshold_diffs = [self.superior_threshold - self.inferior_threshold]
    
    w1 = self.get_imgdistribution("w1.jpeg", self.train.iloc[-self.size_window:, :-1].copy())
    lastupdate = 0
    
    for i in range(len(self.test)):
        print('IBDD -> Example {}/{} drifts: {}'.format(i+1, len(self.test), drift_points), end='\r')
        new_instance = self.test.iloc[i, :]
        
        self.add_instance(new_instance) # incrementing one instance at window
        
        if len(self.tw) >= 10:
            vet_accs = self.apply_quantification(new_instance, vet_accs)

            if len(self.tw) == self.size_window:
                self.append_proportion()
                
                w2 = self.get_imgdistribution("w2.jpeg", self.tw.copy())
                self.nrmse.append(mean_squared_error(w1, w2))
                
                is_drift = self.detect_drift(i)
                if is_drift:
                    drift_points.append(i)
                    self.train = self.tw.copy(deep=True)
                    self.train["class"] = self.tw_labels
                    self.model.fit(self.train.iloc[:, :-1], self.train.iloc[:, -1])
                    self.tw = pd.DataFrame()
                    self.tw_labels = []

        else:
            vet_accs["IBDD"].append(self.model.predict(new_instance.to_frame().T.iloc[:, :-1]).astype(int)[0])
        
    for f in self.files2del:
      os.remove(f"{self.ibdd_dir}/{f}")    
        
    drift_points = {"IBDD": drift_points}
    return pd.DataFrame(vet_accs), drift_points, self.tw_proportions


  def detect_drift(self, index : int) -> bool:
    """
    Make a statistical test with the train window, and the actual window
    The train window is set at the initializer as train, and the actual window is set as tw, when a drift is detected, the tw becomes the new train window and the old one is discarted
    
    Parameters
    ----------
    index : int
        index of iteration
    
    Returns
    -------
    bool : bool
         in case that the statistical test negates the null hypothesis, returns True, other case, False
    """
    drift = False
    if (index - self.lastupdate > 60):
      self.superior_threshold = np.mean(self.nrmse[-50:]) + 2 * np.std(self.nrmse[-50:])
      self.inferior_threshold = np.mean(self.nrmse[-50:]) - 2 * np.std(self.nrmse[-50:])
      self.threshold_diffs.append(self.superior_threshold - self.inferior_threshold)
      self.lastupdate += 1
      
        
    if (all(i >= self.superior_threshold for i in self.nrmse[-self.consecutive_values:])):
      self.superior_threshold = self.nrmse[-1] + np.std(self.nrmse[-50:-1])
      self.inferior_threshold = self.nrmse[-1] - np.mean(self.threshold_diffs)
      self.threshold_diffs.append(self.superior_threshold - self.inferior_threshold)
      self.lastupdate += 1
      drift = True

    elif (all(i <= self.inferior_threshold for i in self.nrmse[-self.consecutive_values:])):
      self.inferior_threshold = self.nrmse[-1] - np.std(self.nrmse[-50:-1])
      self.superior_threshold = self.nrmse[-1] + np.mean(self.threshold_diffs)
      self.threshold_diffs.append(self.superior_threshold - self.inferior_threshold)
      self.lastupdate += 1
      drift=True
      
    return drift


  def find_initial_threshold(self, X_train : object, window_length : int, n_runs : int) -> object :
    """
    Find the initial threshold  that will be used as parameter to determine if the train differs from the current window
    
    Parameters
    ----------
    X_train : object
        train dataframe
    window_length : int
        size of the window
    n_runs : int
        number of iterations to get the threshold
        
    Returns
    -------
    w : object
        a image of the distribution
    
    """
    
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

  def get_imgdistribution(self, name_file : str, data : object) -> object:
    """
    Gets the distribution of a image

    Parameters
    ----------
    name_file : str
        name of the file to get the distribution    
    data : object
        data to get the distribution

    Returns
    -------
    w : object
        the image of distributions
    """
    plt.imsave(f"{self.ibdd_dir}/{name_file}", data.transpose(), cmap = 'Greys', dpi=100)
    w = imread(f"{self.ibdd_dir}/{name_file}")
    return w