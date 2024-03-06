import pandas as pd
import numpy as np

class Experiment:
    
    def __init__(self, train_data, test_data, window_length, model, detector, quantifiers):
        self.trainX = train_data.iloc[:, :-1]
        self.testX = test_data.iloc[:, :-1]
        self.trainY = train_data.iloc[:, -1]
        self.testY = test_data.iloc[:, -1]
        self.window_length = window_length if window_length < len(self.trainY) else len(self.trainY)
        self.model = model.fit(self.trainX, self.trainY)
        self.detector = detector
        self.quantifiers = quantifiers
        self.drifts = []
        
    def run_stream(self):
        window = self.trainX.iloc[-self.window_length:].copy()
        window_labels = self.trainY.iloc[-self.window_length:].copy()
        scores = self.model.predict_proba(window)[:, 1].tolist()
        vet_accs = {}
        iq = 0
        
        for i in range(0, len(self.testY)):
            window = pd.concat([window, self.testX.iloc[[i]]], ignore_index=True).iloc[1:]
            window_labels = pd.concat([window_labels, self.testY.iloc[[i]]], ignore_index=True).iloc[1:]
            
            new_instance_score = self.model.predict_proba(self.testX.iloc[[i]])[:, 1][0]
            scores.append(new_instance_score)
            
            self.detector.Increment(self.testX.iloc[[i]])

            if (iq >= 10):
                vet_accs = self.apply_quantification(self.testX.iloc[[i]], vet_accs)
            if (self.detector.Test()):
                self.drifts.append(i)
                self.trainX = window
                self.trainY = window_labels
                self.model.fit(window, window_labels)
                self.detector.Update()
                iq = 0
            iq += 1
    
    def apply_quantification(self, scores, windowX, new_instance_score):
        app = ApplyQtfs(self.trainX, self.trainY, windowX, self.model, 0.5)
        app.check_train(self.trainX, self.trainY)
        proportions = app.aplly_qtf()