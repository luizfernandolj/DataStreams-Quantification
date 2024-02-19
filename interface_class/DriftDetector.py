from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
from quantifiers.ApplyQtfs import ApplyQtfs
from timeit import default_timer as timer
from sklearn.metrics import accuracy_score
import os

class DriftDetector(ABC):
  """
  A base class to represent all the drift detectors for datastreams. The class simulates a sliding window on a pandas dataframe, predict the scores with a machine learning model, and apply quantification into it
  
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
  tw : pandas.core.frame.DataFrame
      the current window, not containing labels
  tw_labels : list[int]
      the current labels of the window tw
  tw_proportions : list[float]
      class proportion of each window
  
  Methods
  -------
  run_sliding_window():
      is a abstract method that every drift detector need to have. It simulate the instances coming one at a time
  apply_quantification(new_instance) -> dict:
      Apply quantification into the sliding window and tries to improve accuracy of the model and returning the results
  add_instance(new_instance) -> None:
      add the new instance to tw
  append_proportion() -> None:
      append the proportion of the acutal window to the ts_proportions attribute
  detect_drift(*args : any) -> bool
      abstract method that makes a statistical test to detect if a drift has occured
  """
  
  def __init__(self, train, test, size_window, model):
    """
    Construct all the necessacy attributes for each type of DriftDetector

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
        tw : pandas.core.frame.DataFrame
            the current window, not containing labels
        tw_labels : list[int]
            the current labels of the window tw
        tw_proportions : list[float]
            class proportion of each window
    
    """
    self.train = train
    self.test = test
    self.size_window : int = size_window
    self.model = model.fit(train.iloc[:, :-1], train.iloc[:, -1])
    self.tw : object = pd.DataFrame()
    self.tw_labels : list[int] = []
    self.tw_proportions : list[float] = []
  
  
  @abstractmethod  
  def run_sliding_window(self) -> list:
    """
    Method to simulate instances coming one at a time, from a DataStream, applying different statistical tests for each detector 
    
    Returns
    -------
    vet_accs : pandas.core.frame.DataFrame
        predictions of the model along all datastream, with the quantification predictions
    drift_points : dict[str:list[int]]
        positions where the drift detector detected a drift
    tw_proportions = list[float]
        proportions of all windows 
    """
    pass
  
  
  def append_proportion(self) -> None:
    """
    Append the positive class proportion of the window if length of tw is equals to size_window
    """
    if len(self.tw) == self.size_window:
      total_size : int = len(self.tw_labels)
      pos_prop : float = round(total_size / sum(self.tw_labels), 2) # computing positive class proportion
      self.tw_proportions.append(pos_prop)
      
      
  def add_instance(self, new_instance : object) -> None:
    """
    Add the new instance to the window (tw), as long to the window labels (tw_labels)
    If length of the window is equals to size_window then exclude the first element of window
     
    Parameters  
    ----------
    new_instance : object
        the current instance of the datastream to be putted into the window and window labels
    """
    self.tw = pd.concat([self.tw, new_instance.iloc[:-1].to_frame().T], ignore_index=True)
    self.tw_labels.append(new_instance.iloc[-1])
    if len(self.tw) == self.size_window+1:
      self.tw = self.tw[1:]
      self.tw_labels = self.tw_labels[1:]
      
      
  def apply_quantification(self, new_instance : object, vet_accs : dict) -> dict:
    """
    Apply quantification methods to the window to find the positive class proportion and use it to compute the threshold and classify the new instance
    
    Parameters
    ----------
    new_instance : object
        the current instance of the datastream
    vet_accs : dict[str : list[int]]
        dictionary of the predicted classes predicted by the model, with the predicted classes applying quantification
    
    Returns
    -------
    vet_accs : dict[str : list[int]]
        dictionary of the predicted classes predicted by the model, with the predicted classes applying quantification (after applying quantification into that new instance)
    """
    first_key = list(vet_accs.keys())[0]
    score = self.model.predict_proba(new_instance.to_frame().T.iloc[:, :-1])[:, 1]
    
    #                 trainX                     trainy              window  classifier  threshold        
    app = ApplyQtfs(self.train.iloc[:, :-1], self.train.iloc[:, -1], self.tw, self.model, 0.5)
    proportions = app.aplly_qtf() # getting the proportions of each quantifier
    
    for qtf, proportion in proportions.items():
        name = f"{first_key}-{qtf}"
        
        pos_scores = self.model.predict_proba(self.tw)[:,1].tolist() # predicting the 'probabilities' of the window
        thr = app.calc_threshold(proportion, pos_scores) # getting the threshold using the positive proportion
        #print(f"qtf{name} - proportion{proportion} - threshold{thr}")
        if name not in vet_accs:
          vet_accs[name] = []
        if len(self.tw) == 10:
            vet_accs[name].extend(vet_accs[first_key][-9:])
        vet_accs[name].append(1 if score >= thr else 0)
        # Adding the predicted instance to the first key of the dictionary, which is the predict classes without quantification
    vet_accs[first_key].append(self.model.predict(new_instance.to_frame().T.iloc[:, :-1]).astype(int)[0])
    print(pd.DataFrame(vet_accs)) 
    
    return vet_accs
  
  
  @abstractmethod
  def detect_drift(*args : any) -> bool:
    """
    Abstract class that every drift detector need to have, it will make a statistical test with the train window, and the actual window
    The train window is set at the initializer as train, and the actual window is set as tw, when a drift is detected, the tw becomes the new train window and the old one is discarted
    
    Parameters
    ----------
    *args : any
        the value to make the statistical test, every detector has a diferent value
    
    Returns
    -------
    bool : bool
         in case that the statistical test negates the null hypothesis, returns True, other case, False
    """
    pass