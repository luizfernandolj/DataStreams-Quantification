import numpy as np
import pandas as pd

import time


def MS_method2(test_scores, tprfpr):
    """Median Sweep2

    It quantifies events based on their scores, applying Median Sweep (MS2) method, according to Forman (2006).
    
    Parameters
    ----------
    test scores: array
        A numeric vector of scores predicted from the test set.
    TprFpr : matrix
        A matrix of true positive (tpr) and false positive (fpr) rates estimated on training set, using the function getScoreKfolds().
        
    Returns
    -------
    array
        the class distribution of the test. 
    """
    start = time.time()

    index = np.where(abs(tprfpr['tpr'] - tprfpr['fpr']) >(1/4) )[0].tolist()
    if index == 0:
        index = np.where(abs(tprfpr['tpr'] - tprfpr['fpr']) >=0 )[0].tolist()

    
    prevalances_array = []    
    for i in index:
        
        threshold, fpr, tpr = tprfpr.loc[i]
        estimated_positive_ratio = len(np.where(test_scores >= threshold)[0])/len(test_scores)
        
        diff_tpr_fpr = abs(float(tpr-fpr))  
    
        if diff_tpr_fpr == 0.0:            
            diff_tpr_fpr = 1     
    
        final_prevalence = abs(estimated_positive_ratio - fpr)/diff_tpr_fpr
        
        prevalances_array.append(final_prevalence)  
  
    pos_prop = np.median(prevalances_array)
    
    if pos_prop <= 0:                           #clipping the output between [0,1]
        pos_prop = 0
    elif pos_prop >= 1:
        pos_prop = 1
    else:
        pos_prop = pos_prop
    stop = time.time()
    #return stop - start
    return pos_prop
