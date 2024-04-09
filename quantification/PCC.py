import numpy as np
import time
def PCC(calib_clf,test_data, thr = 0.5):
    """Probabilistic Classify & Count (PCC)

    It quantifies events based on Calibrated classifier and correct the estimate using TPR and FPR, applying Probabilistic Classify & Count (PCC) method, according to Bella (2010).
    
    Parameters
    ----------
    calib_clf : Object
        Calibrated classifier previously trained from some training set partition.
    Test_data : Dataframe 
        A DataFrame of the test data. 
    thr : float  
        The threshold value for hard predictions. Default value = 0.5.
    
    Returns
    -------
    array
        the class distribution of the test. 
    """

    start = time.time()
    calibrated_predictions = calib_clf.predict_proba(test_data)[:,1]    
    pos_prop = np.mean(calibrated_predictions)
    stop = time.time()
    #return stop - start
    return  pos_prop
