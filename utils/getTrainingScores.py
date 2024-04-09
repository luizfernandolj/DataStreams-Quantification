import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold



def getTrainingScores(X_train, Y_train, folds, clf):
    """
    This function estiamtes the scores of the provided training set using k-fold stratified cross-validation
    
    Parameters
    -----------
    X_train :  dataframe
        A dataframe of training data.
    Y_train : vector
        It contains the label information of trainig data.
    folds : integer
        number of folds 
    clf : classifier model.

    Returns
    ---------
    Dataframe
        estimated scores.
    """
    
    skf = StratifiedKFold(n_splits=folds)    
    results = []
    class_labl = []
    
    for fold_i, (train_index,valid_index) in enumerate(skf.split(X_train,Y_train)):
        print('  Fold #%d' % (fold_i + 1))
        print('Training_len', len(X_train))
        
        tr_data = pd.DataFrame(X_train.iloc[train_index])   #Train data and labels
        tr_lbl = Y_train.iloc[train_index]
        
        valid_data = pd.DataFrame(X_train.iloc[valid_index])  #Validation data and labels
        valid_lbl = Y_train.iloc[valid_index]
        
        clf.fit(tr_data, tr_lbl)
        
        results.extend(clf.predict_proba(valid_data)[:,1])     #evaluating scores
        class_labl.extend(valid_lbl)
        
        print('SCORES_Length:',len(results))

    # Fitting the final classifier
    clf.fit(X_train, Y_train)
    
    train_scores = np.c_[results,class_labl]
    train_scores = pd.DataFrame(train_scores)
    train_scores.columns = ['scores', 'class']
    train_scores['class'] = np.int0(train_scores['class'])
    
    return train_scores, clf      
