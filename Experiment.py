import pandas as pd
import numpy as np
import pdb
from utils.getTrainingScores import getTrainingScores
from utils.getTPRFPR import getTPRFPR
from utils.applyquantifiers import apply_quantifier
from utils.get_best_thr import get_best_threshold

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
        self.quantifier_methods = ["CC", "ACC", "MS", "DyS"]
        
    def run_stream(self):
        """Simulate a Datastream, running a window and testing the occurrences of drifts. While applying quantification
        """
        # Starting window
        window = self.trainX.iloc[-self.window_length:].copy(deep=True).reset_index(drop=True)
        window_labels = self.trainY.iloc[-self.window_length:].copy(deep=True).reset_index(drop=True)
        
        # Getting the training scores, and the initial things we need to run the quantification methods
        scores, tprfpr, pos_scores, neg_scores = self.get_train_values()
        test_scores = [] 
        
        vet_accs = {self.detector_name : []}
        iq = 0
        
        # Running Datastream
        for i in range(0, len(self.testY)):
            print(f"{self.detector_name}-{i}", end='\r')
            new_instance = self.testX.iloc[[i]]
            # Step of the window
            window = pd.concat([window, new_instance], ignore_index=True).iloc[1:]
            window_labels = pd.concat([window_labels, self.testY.iloc[[i]]], ignore_index=True).iloc[1:]
            
            # Getting the positive score of each instance
            new_instance_score = self.model.predict_proba(new_instance)[:, 1][0]
            test_scores.append(new_instance_score)
            
            # Incrementing the new instance to the detector (IKS, IBDD and WRS)
            self.detector.Increment(self.testX.loc[i], window, i)

            if (iq >= self.window_length-1):
                # Applying quantification after 10 instances 
                #pdb.set_trace()
                vet_accs = self.apply_quantification(scores,
                                                     np.array(test_scores),
                                                     tprfpr,
                                                     pos_scores,
                                                     neg_scores,
                                                     window, 
                                                     new_instance_score, 
                                                     vet_accs)
                scores = scores[1:]
            else:
                if len(vet_accs) > 2:       
                    for key, p in vet_accs.items():
                        if key != self.detector_name:             
                            vet_accs[key].append(self.model.predict(new_instance)[0])
              
                
            vet_accs[self.detector_name].append(self.model.predict(new_instance)[0])
            
                
            if (self.detector.Test(i)): # Statistical test if the drift occured
                print("drift")
                self.drifts.append(i)
                
                # turning current window into train, and updating classifier and detector
                self.trainX = window 
                self.trainY = window_labels
                scores, tprfpr, pos_scores, neg_scores = self.get_train_values()
                self.detector.Update(window)
                iq = -1
            iq += 1
    
        return pd.DataFrame(vet_accs), {self.detector_name:self.drifts}
    
    def apply_quantification(self,
                             scores: object,
                             test_scores: list[float],
                             tprfpr : object,
                             pos_scores : object,
                             neg_scores : object,
                             windowX : object, 
                             new_instance_score : float,
                             vet_accs : dict[str: list[int]]):
        """Apply quantification into window, getting the positive scores and fiding the best threshold to classify the new instance

        Args:
            scores (list[float]): positive scores of the window, predicted by classifier
            windowX (Any): current window without labels
            new_instance_score (Any): score of the new instance
            vet_accs (dict[str: list[int]]) : dictionary containing the predicted class of each quantification algorithm and also the classification only
        """
        proportions = {}
        for qtf_method in self.quantifier_methods:
            pred_pos_prop = apply_quantifier(qntMethod=qtf_method,
                                             clf = self.model,
                                             scores=scores['scores'],
                                             p_score=pos_scores,
                                             n_score=neg_scores,
                                             train_labels=scores['class'],
                                             test_score=test_scores,
                                             TprFpr=tprfpr,
                                             thr=0.5,
                                             measure="topsoe",
                                             test_data=windowX)
            proportions[qtf_method] = pred_pos_prop
            
        for qtf, proportion in proportions.items():
            name = f"{self.detector_name}-{qtf}"
            
            thr = get_best_threshold(proportion, test_scores) # getting the threshold using the positive proportion
            
            if name not in vet_accs:
                vet_accs[name] = []
                vet_accs[name].extend(vet_accs[self.detector_name][-self.window_length:])
            
            # Using the threshold to determine the class of the new instance score
            vet_accs[name].append(1 if new_instance_score >= thr else 0)
        return vet_accs
            
            
        
    def get_train_values(self):
        scores, self.model = getTrainingScores(self.trainX, self.trainY, 10, self.model)
        tprfpr = getTPRFPR(scores)
        pos_scores = scores[scores["class"]==1]["scores"]
        neg_scores = scores[scores["class"]==0]["scores"]

        return scores, tprfpr, pos_scores, neg_scores
        