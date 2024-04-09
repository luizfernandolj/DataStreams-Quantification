import pandas as pd
import numpy as np

def getTPRFPR(scores):
    """getTPRFPR

    This function receives the training scores and estimate the trupoe positive rate(TPR) and false positive rate(FPR) matrix.
    
    Parameters
    ----------
    scores : array
        The numeric array of scores obtaining from validation set using 10-fold stratified cross-validation
      
    Returns
    -------
    Matrix
        It returns a matrix of TPR and FPR. 
    """
    unique_scores = np.linspace(0,1,101)
        
    TprFpr = pd.DataFrame(columns=['threshold','fpr', 'tpr'])
    total_positive = len(scores[scores['class']==1])
    total_negative = len(scores[scores['class']==0])  
    for threshold in unique_scores:
        fp = len(scores[(scores['scores'] > threshold) & (scores['class']==0)])  
        tp = len(scores[(scores['scores'] > threshold) & (scores['class']==1)])

        tpr = round(tp/total_positive,2)
        fpr = round(fp/total_negative,2)
    
        aux = pd.DataFrame([[threshold, fpr, tpr]])
        aux.columns = ['threshold', 'fpr', 'tpr']    
        TprFpr = pd.concat([TprFpr, aux])
     
    return TprFpr
