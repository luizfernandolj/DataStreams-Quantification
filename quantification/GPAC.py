import cvxpy as cvx
import numpy as np
import pdb
import time

def GPAC(train_scores, test_scores, train_labels, nclasses):
    
    CM = np.zeros((nclasses, nclasses))
    for i in range(nclasses):
        idx = np.where(train_labels == i)[0]
        CM[i] = np.sum(train_scores[idx], axis=0)
        CM[i] /= np.sum(CM[i])
    CM = CM.T
    start = time.time()
    p_y_hat = np.sum(test_scores, axis = 0)
    p_y_hat = p_y_hat / np.sum(p_y_hat)

    p_hat = cvx.Variable(CM.shape[1])
    constraints = [p_hat >= 0, cvx.sum(p_hat) == 1.0]
    problem = cvx.Problem(cvx.Minimize(cvx.norm(CM @ p_hat - p_y_hat)), constraints)
    problem.solve()
    stop = time.time()
    #return stop - start
    return p_hat.value[1]
