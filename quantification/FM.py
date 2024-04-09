import numpy as np
import cvxpy as cvx

import pdb

def FM(train_scores, test_scores, train_labels, nclasses):
    
    CM = np.zeros((nclasses, nclasses))
    y_cts = np.array([np.count_nonzero(train_labels == i) for i in range(nclasses)])
    p_yt = y_cts / train_labels.shape[0]
    
    for i in range(nclasses):        
        idx = np.where(train_labels == i)[0]
        CM[:, i] += np.sum(train_scores[idx] > p_yt, axis=0) 
    CM = CM / y_cts
    p_y_hat = np.sum(test_scores > p_yt, axis = 0) / test_scores.shape[0]
    
    p_hat = cvx.Variable(CM.shape[1])
    constraints = [p_hat >= 0, cvx.sum(p_hat) == 1.0]
    problem = cvx.Problem(cvx.Minimize(cvx.norm(CM @ p_hat - p_y_hat)), constraints)
    problem.solve()
    return p_hat.value[1]