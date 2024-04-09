import numpy as np
from sklearn.metrics import confusion_matrix
import cvxpy as cvx
import time
import pdb

def GAC(train_scores, test_scores, train_labels, nclasses):
    start = time.time()
    yt_hat = np.argmax(train_scores, axis = 1)
    y_hat = np.argmax(test_scores, axis = 1)
    CM = confusion_matrix(train_labels, yt_hat, normalize="true").T
    p_y_hat = np.zeros(nclasses)
    values, counts = np.unique(y_hat, return_counts=True)
    p_y_hat[values] = counts
    p_y_hat = p_y_hat/p_y_hat.sum()
    
    p_hat = cvx.Variable(CM.shape[1])
    constraints = [p_hat >= 0, cvx.sum(p_hat) == 1.0]
    problem = cvx.Problem(cvx.Minimize(cvx.norm(CM @ p_hat - p_y_hat)), constraints)
    problem.solve()

    stop = time.time()
    #return stop - start
    return p_hat.value[1]
