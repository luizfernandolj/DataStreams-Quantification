import numpy as np
import time
def T50(test_scores, tprfpr):
    """T50 method

    It quantifies events based on threshold selection criteria (tpr = 50%), applying T50 method, according to Forman (2006).
    
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
    
    index = np.abs(tprfpr['tpr'] - 0.5).idxmin()      #taking threshold tpr and fpr where tpr=50% or (tpr - 50%) is minimum
   
    threshold, fpr, tpr = tprfpr.loc[index]            
    
    start = time.time()

    class_prop = len(np.where(test_scores >= threshold)[0])/len(test_scores)

    #pos_prop = round(abs(class_prop - fpr)/abs(tpr - fpr),2)   #adjusted class proportion
    
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
    return stop - start
    return pos_prop
    