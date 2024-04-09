import numpy as np
import time

def classify_count(test_scores,thr=0.5):
    """Classify & Count (CC)

    It quantifies events based on trained classifier, applying Classify & Count (ACC) method, according to Forman (2005).
    
    Parameters
    --------
    Test scores : array
        A numeric vector of scores predicted from the test set. 
    thr : float  
        The threshold value for hard predictions. Default value = 0.5.
    
    Returns
    -------
    array
        the class distribution of the test. 
    """
    start = time.time()
    count = len(np.where(test_scores >= thr)[0])  
    pos_prop = count/len(test_scores)
    stop = time.time()
    #return stop - start
    return pos_prop
