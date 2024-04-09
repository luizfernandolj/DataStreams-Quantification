import numpy as np
import time

def SMM(pos_scores, neg_scores,test_scores):
    """Sample Mean Matching

    It quantifies events based on Arithmetic mean value of positive, negative and test scores.
    
    Parameters
    ----------
    Positive scores : array
        A numeric vector of positive scores estimated from the validation set using 10-fold stratified cross-validation.
    Negative scores : array
        A numeric vector of Negative scores estimated from the validation set using 10-fold stratified cross-validation.
    Test scores : array
        A numeric vector of scores predicted from the test set.
       
    Returns
    -------
    array
        the class distribution of the test. 
    """
    
    mean_pos_scr = np.mean(pos_scores)
    mean_neg_scr = np.mean(neg_scores)  #calculating mean of pos & neg scores
    
    start = time.time()
    mean_te_scr = np.mean(test_scores)              #Mean of test scores
         
    alpha =  (mean_te_scr - mean_neg_scr)/(mean_pos_scr - mean_neg_scr)     #evaluating Positive class proportion
      
    if alpha <= 0:   #clipping the output between [0,1]
        pos_prop = 0
    elif alpha >= 1:
        pos_prop = 1
    else:
        pos_prop = alpha
    
    stop = time.time()
    #return stop - start

    return pos_prop
    
