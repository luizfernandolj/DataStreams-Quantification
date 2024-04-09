from quantification.CC import classify_count
from quantification.ACC import ACC
from quantification.PCC import PCC
from quantification.PACC import PACC
from quantification.HDy import Hdy
from quantification.X import X
from quantification.MAX import Max 
from quantification.dys_method import dys_method
from quantification.sord import SORD_method
from quantification.MS import MS_method
from quantification.T50 import T50
import numpy as np
import pandas as pd

def apply_quantifier(qntMethod, 
                    clf,
                    scores, 
                    p_score, 
                    n_score, 
                    train_labels, 
                    test_score, 
                    TprFpr, 
                    thr, 
                    measure,                     
                    test_data):
    """This function is an interface for running different quantification methods.
 
    Parameters
    ----------
    qntMethod : string
        Quantification method name
    p_score : array
        A numeric vector of positive scores estimated either from a validation set or from a cross-validation method.
    n_score : array
        A numeric vector of negative scores estimated either from a validation set or from a cross-validation method.
    test : array
        A numeric vector of scores predicted from the test set.
    TprFpr : matrix
        A matrix of true positive (tpr) and false positive (fpr) rates estimated on training set, using the function getScoreKfolds().
    thr : float
        The threshold value for classifying and counting. Default is 0.5.
    measure : string
        Dissimilarity function name used by the DyS method. Default is "topsoe".
    calib_clf : object
        A calibrated classifier used when PCC or PACC methods are called by the main experimental setup.
    te_data : dataframe
        A dataframe of test data
    Returns
    -------
    array
        the class distribution of the test calculated according to the qntMethod quantifier. 
    """
    
    if qntMethod == "CC":
        return classify_count(test_score, thr)
    if qntMethod == "ACC":        
        return ACC(test_score, TprFpr)   
    if qntMethod == "HDy":
        return Hdy(p_score, n_score, test_score)
    if qntMethod == "DyS":
        return dys_method(p_score, n_score, test_score,measure)
    if qntMethod == "SORD":
        return SORD_method(p_score, n_score,test_score)
    if qntMethod == "MS":
        return MS_method(test_score, TprFpr)
    if qntMethod == "MAX":
        return Max(test_score, TprFpr)
    if qntMethod == "X":
        return X(test_score, TprFpr)
    if qntMethod == "T50":
        return T50(test_score, TprFpr)
    if qntMethod == "PCC":
        return PCC(clf, test_data,thr)
    if qntMethod == "PACC":
        return PACC(clf, test_data, TprFpr, thr)
