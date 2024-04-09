import numpy as np
import time

def X(test_scores, tprfpr):
    """X method

    It quantifies events based on threshold selection criteria (1 - tpr = fpr), applying X method, according to Forman (2006).
    
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

    min_index = (np.abs((1 - tprfpr['tpr']) - tprfpr['fpr'])).idxmin()
    threshold, fpr, tpr = tprfpr.loc[min_index]            #taking threshold,tpr and fpr where [(1 -tpr) - fpr] is minimum
    start = time.time()
    class_prop = len(np.where(test_scores >= threshold)[0])/len(test_scores)
    
    if (tpr - fpr) == 0:
        pos_prop = class_prop
    else:
        pos_prop = (class_prop - fpr)/(tpr - fpr)   #adjusted class proportion

    if pos_prop <= 0:                           #clipping the output between [0,1]
        pos_prop = 0
    elif pos_prop >= 1:
        pos_prop = 1
    else:
        pos_prop = pos_prop
    stop = time.time()
    #return stop - start
    return pos_prop
    