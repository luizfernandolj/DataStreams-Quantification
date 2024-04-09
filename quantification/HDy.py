import numpy as np
import utils.DyS_utils as qntu
import time


def Hdy(pos_scores, neg_scores,test_scores):
    """HDy

    It quantifies events based on Hellinger distance between the mixture of positive and negative scores and the test scores, applying HDy method, according to Gonzalez Castro(2013).
    
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
    
    bin_size = np.linspace(10,110,11)       #creating bins from 10 to 110 with step size 10
    #alpha_values = [round(x, 2) for x in np.linspace(0,1,101)]
    alpha_values = np.linspace(0,1,101)
    
    result = []
    num_bins = []
    start = time.time()
    for bins in bin_size:
        
        p_bin_count = qntu.getHist(pos_scores, bins)
        n_bin_count = qntu.getHist(neg_scores, bins)
        te_bin_count = qntu.getHist(test_scores, bins)

        vDist = []
        
        for x in alpha_values:
            x= np.round(x,2)
            vDist.append(qntu.DyS_distance(((p_bin_count*x) + (n_bin_count*(1-x))), te_bin_count, measure="hellinger"))

        result.append(alpha_values[np.argmin(vDist)])
        pos_prop = np.median(result)
        
        #num_bins.append(bins)
    #bin_proportion = pd.concat([pd.DataFrame(num_bins), pd.DataFrame(result)], axis=1)
    #bin_proportion.columns = ["bins","class_proportion"]
    stop = time.time()
    #return stop - start
    return pos_prop
    


    
        




