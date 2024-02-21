from interface_class.DriftDetector import DriftDetector

import pandas as pd
import numpy as np
from scipy import stats



class WRS(DriftDetector):
  """
  The WRS () class is a drift detector to detect if the data has changed over time, inheriting from DriftDetector class
  
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
  

  def __init__(self, train, test, size_window, model, threshold):
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
    self.threshold = threshold
    self.flag = False

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
    vet_accs : dict[str:list[int]] = {"WRS": []}
    
    w1 = self.train.iloc[-self.size_window:, :-1].copy()
    
    for i in range(len(self.test)):
        print('Example {}/{} drifts: {}'.format(i+1, len(self.test), drift_points), end='\r')
        new_instance = self.test.loc[i]
        
        self.add_instance(new_instance) # incrementing one instance at window
        
        if len(self.tw) >= 10:
            vet_accs = self.apply_quantification(new_instance, vet_accs)

            if len(self.tw) == self.size_window:
                self.append_proportion()
                
                w2 = self.tw.copy()
                _, n_features = self.tw.shape
                
                is_drift = self.detect_drift(n_features, w1, w2)
                if is_drift:
                    print('drift')
                    drift_points.append(i)
                    self.train = self.tw.copy(deep=True)
                    self.train["class"] = self.tw_labels
                    self.tw = pd.DataFrame()
                    self.tw_labels = []
                    self.model.fit(self.train.iloc[:, :-1], self.train.iloc[:, -1])

        else:
            vet_accs["WRS"].append(self.model.predict(new_instance.to_frame().T.iloc[:, :-1]).astype(int)[0])
    
    drift_points = {"WRS": drift_points}
    return pd.DataFrame(vet_accs), drift_points, self.tw_proportions


  def detect_drift(self, n_features : int, w1 : object, w2 : object) -> bool:
    """
    Make a statistical test with the train window, and the actual window
    The train window is set at the initializer as train, and the actual window is set as tw, when a drift is detected, the tw becomes the new train window and the old one is discarted
    
    Parameters
    ----------
    n_features
        number of features of the window
    w1 : object
        train for comparation with the window
    w2 : object
        current window for comparation with w1 (train)
    
    Returns
    -------
    bool : bool
         in case that the statistical test negates the null hypothesis, returns True, other case, False
    """
    self.flag = False
    for j in range(0, n_features):
      _, p_value = stats.ranksums(w1.iloc[:,j], w2.iloc[:,j])        
      if (p_value <= self.threshold):
          self.flag = True

    if self.flag:
      w1 = w2 # update the reference window with recent data of w2
      
    return self.flag