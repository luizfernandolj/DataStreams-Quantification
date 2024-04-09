import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
import pandas as pd

def getCalibratedTrainingScores(X_train, Y_train, folds, model):
   
    nclasses = len(np.unique(Y_train))
    train_scores = np.zeros((len(X_train), nclasses)) 
    train_labels = np.zeros(X_train.shape[0]) 
       
    Y_cts = np.unique(Y_train, return_counts=True)
    nfolds = min(folds, min(Y_cts[1]))
    
    if nfolds > 1:
        kfold = StratifiedKFold(n_splits=nfolds, random_state=1, shuffle=True)
        for train, test in kfold.split(X_train, Y_train):
            X_trn, x_val, y_trn, y_val = train_test_split(X_train.iloc[train], Y_train.iloc[train], test_size = 0.2, stratify=Y_train.iloc[train])
            model.fit(X_trn, y_trn)
            calib_clf = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
            calib_clf.fit(x_val, y_val)
            train_scores[test] = calib_clf.predict_proba(X_train)[test]
            train_labels[test] = Y_train.iloc[test]
    
    X_trn, x_val, y_trn, y_val = train_test_split(X_train, Y_train, test_size = 0.2, stratify=Y_train)
    model.fit(X_trn, y_trn)
    calib_clf = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
    calib_clf.fit(x_val, y_val)   
    
    if nfolds < 2:
        train_scores = calib_clf.predict_proba(X_train)

    train_scores = np.c_[train_scores[:,1],train_labels]
    train_scores = pd.DataFrame(train_scores)
    train_scores.columns = ['scores', 'class']

    return train_scores, calib_clf