import numpy as np
import time

def PACC(calib_clf, test_data, tprfpr, thr = 0.5):

    """Probabilistic Adjust Classify & Count (PACC)

    It quantifies events based on Calibrated classifier and correct the estimate using TPR and FPR, applying Probabilistic Adjust Classify & Count (PACC) method, according to Bella (2010).
    
    Parameters
    ----------
    calib_clf : Object/Model
        Calibrated classifier trained from the training set partition.
    TprFpr : matrix
        A matrix of true positive (tpr) and false positive (fpr) rates estimated on training set, using the function getScoreKfolds().
    Test_data : Dataframe 
        A DataFrame of the test data.
    thr : float  
        The threshold value for hard predictions. Default value = 0.5.
    
    Returns
    -------
    array
        the class distribution of the test. 
    """

    TprFpr = tprfpr[tprfpr['threshold'] == thr]
    start = time.time()
    calibrated_predictions = calib_clf.predict_proba(test_data)[:,1]
    pos_prop = np.mean(calibrated_predictions)    
    diff_tpr_fpr = (float(TprFpr['tpr']) - float(TprFpr['fpr']))
    pos_prop = (pos_prop - float(TprFpr['fpr'])) / diff_tpr_fpr

    if pos_prop <= 0:                           #clipping the output between [0,1]
        pos_prop = 0
    elif pos_prop >= 1:
        pos_prop = 1
    else:
        pos_prop = pos_prop

    stop = time.time()
    #return stop - start
    return pos_prop
    
