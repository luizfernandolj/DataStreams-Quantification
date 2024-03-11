import pandas as pd
import numpy as np
import pdb
from quantifiers.ApplyQtfs import ApplyQtfs

class Experiment:
    
    def __init__(self, train_data, test_data, window_length, model, detector, detector_name):
        self.trainX = train_data.iloc[:, :-1]
        self.testX = test_data.iloc[:, :-1]
        self.trainY = train_data.iloc[:, -1]
        self.testY = test_data.iloc[:, -1]
        self.window_length = window_length if window_length < len(self.trainY) else len(self.trainY)
        self.model = model.fit(self.trainX, self.trainY)
        self.detector = detector
        self.detector_name = detector_name
        self.drifts = []
        self.app = ApplyQtfs(self.trainX, self.trainY, self.model, 0.5)
        
    def run_stream(self):
        """Simulate a Datastream, running a window and testing the occurrences of drifts. While applying quantification
        """
        # Starting window
        window = self.trainX.iloc[-self.window_length:].copy(deep=True).reset_index(drop=True)
        window_labels = self.trainY.iloc[-self.window_length:].copy(deep=True).reset_index(drop=True)
        
        scores = self.model.predict_proba(window)[:, 1].tolist() # Getting the positive scores of the start window
        vet_accs = {'real': [], self.detector_name : []}
        iq = 0
        
        # Running Datastream
        for i in range(0, len(self.testY)):
            print(i)
            new_instance = self.testX.iloc[[i]]
            # Step of the window
            window = pd.concat([window, new_instance], ignore_index=True).iloc[1:]
            window_labels = pd.concat([window_labels, self.testY.iloc[[i]]], ignore_index=True).iloc[1:]
            
            # Getting the positive score of each instance
            new_instance_score = self.model.predict_proba(new_instance)[:, 1][0]
            scores.append(new_instance_score)
            
            # Incrementing the new instance to the detector (IKS, IBDD and WRS)
            self.detector.Increment(self.testX.iloc[[i]], window, i)

            if (iq >= 10):
                # Applying quantification after 10 instances 
                #pdb.set_trace()
                vet_accs = self.apply_quantification(scores, 
                                                     window, 
                                                     new_instance_score, 
                                                     vet_accs)
            vet_accs[self.detector_name].append(self.model.predict(new_instance)[0])
            vet_accs['real'].append(self.testY.iloc[[i]].tolist()[0])
                
            if (self.detector.Test(i)): # Statistical test if the drift occured
                print("drift")
                self.drifts.append(i)
                
                # turning current window into train, and updating classifier and detector
                self.trainX = window 
                self.trainY = window_labels
                self.model.fit(window, window_labels)
                self.detector.Update(window)
                iq = 0
            iq += 1
    
        return pd.DataFrame(vet_accs), self.drifts
    
    def apply_quantification(self, pos_scores: list[float], windowX : object, new_instance_score : float, vet_accs : dict[str: list[int]]):
        """Apply quantification into window, getting the positive scores and fiding the best threshold to classify the new instance

        Args:
            scores (list[float]): positive scores of the window, predicted by classifier
            windowX (Any): current window without labels
            new_instance_score (Any): score of the new instance
            vet_accs (dict[str: list[int]]) : dictionary containing the predicted class of each quantification algorithm and also the classification only
        """
        self.app.check_train(self.trainX, self.trainY)
        proportions : dict[str:float]= self.app.aplly_qtf(windowX) # positive proportion of each quantifier
        print("real prop", sum(vet_accs['real'])/len(vet_accs['real']))
        for qtf, proportion in proportions.items():
            name = f"{self.detector_name}-{qtf}"
            print(qtf, proportion)
            
            thr = self.app.get_best_threshold(proportion, pos_scores) # getting the threshold using the positive proportion
            
            if name not in vet_accs:
                vet_accs[name] = []
            #pdb.set_trace()
            if len(vet_accs[self.detector_name])-1 > len(vet_accs[name]):
                vet_accs[name].extend(vet_accs[self.detector_name][-10:])
            # Using the threshold to determine the class of the new instance score
            vet_accs[name].append(1 if new_instance_score >= thr else 0)
        return vet_accs
            
            
        
        
        