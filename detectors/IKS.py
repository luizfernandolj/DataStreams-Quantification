from interface_class.DriftDetector import DriftDetector

import pandas as pd
import numpy as np
from ikscode.IKSSW import IKSSW

class IKS(DriftDetector):
  """
  The IKS () class is a drift detector to detect if the data has changed over time, inheriting from DriftDetector class
  
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
  ca : float
      #TODO
      
  Methods
  -------
  run_sliding_window():
      Simulates the instances coming one at a time
  detect_drift(ca : float) -> bool
      abstract method that makes the Kolmogorov-Smirnov test to detect if a drift has occured
  """
  

  def __init__(self, train, test, size_window, model, ca):
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
        ca : float
            #TODO
        ikssw : object
            the IKSSW implementation, to make the Kolmogorov-Smirnov test
    """
    super().__init__(train, test, size_window, model)
    self.ca : float = ca
    self.ikssw = IKSSW(train.iloc[-size_window:, :-1].values.tolist())

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
    vet_accs : dict[str:list[int]] = {"IKS": []}
    
    for i in range(len(self.test)):
        print('IKS -> Example {}/{} drifts: {}'.format(i+1, len(self.test), drift_points), end='\r')
        new_instance = self.test.iloc[i, :]
        
        self.add_instance(new_instance) # incrementing one instance at window
        self.ikssw.Increment(new_instance.values.tolist()) # IKS incrementing one instance and excluding the first
        
        if len(self.tw) >= 10:
            vet_accs = self.apply_quantification(new_instance, vet_accs)

            if len(self.tw) == self.size_window:
                is_drift = self.detect_drift(self.ca)
                self.append_proportion()
                if is_drift:
                    drift_points.append(i)
                    
                    self.ikssw.Update()
                    self.train = self.tw.copy(deep=True)
                    self.train["class"] = self.tw_labels
                    self.tw = pd.DataFrame()
                    self.tw_labels = []
                    self.model.fit(self.train.iloc[:, :-1], self.train.iloc[:, -1])

        else:
            vet_accs["IKS"].append(self.model.predict(new_instance.to_frame().T.iloc[:, :-1]).astype(int)[0])
        
    drift_points = {"IKS": drift_points}
    return pd.DataFrame(vet_accs), drift_points, self.tw_proportions


  def detect_drift(self, ca : float = 1.95) -> bool:
    """
    Make a statistical test with the train window, and the actual window
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
    return self.ikssw.Test(ca)
    





