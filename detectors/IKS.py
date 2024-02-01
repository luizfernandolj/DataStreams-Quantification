from interface_class.DriftDetector import DriftDetector

import pandas as pd
import numpy as np

class IKS(DriftDetector):

  def __init__(self, stream, size_train, size_window, model, context_list, ca):
    super().__init__(stream, size_train, size_window, model, context_list)
    self.ca = ca
    self.t = False

  def runslidingwindow(self):

    #ikssw = IKSSW(self.train_X.iloc[-self.size_window:].values.tolist())
    self.vet_accs["IKS"] = []
    for index, new_instance in self.test.iterrows():
      print('Example {}/{}'.format(index, len(self.test)), end='\r')
      if len(self.tw) >= 10:
        self.get_real_proportion(index)
        self.apply_qtf(new_instance)

        if len(self.tw) == 300:
          self.statistical_test()
          if self.t:
            self.trainX = self.tw
            self.model.fit(self.trainX, self.real_labels_window)
          else:
            # TODO

      else:
        self.vet_accs["IKS"].append(self.model.predict(new_instance.to_frame().T).astype(int)[0])
        self.twlabels = self.vet_accs["IKS"].copy()
        self.tw = pd.concat([self.tw, new_instance.to_frame().T], ignore_index=True)


  def statistical_test(self):
    # TODO
    pass





