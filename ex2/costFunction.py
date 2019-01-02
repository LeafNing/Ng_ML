import numpy as np
from sigmoid import *


def cost_function(theta, X, y):
    m = y.size

    # You need to return the following values correctly
    cost = 0
    grad = np.zeros(theta.shape)

    # ===================== Your Code Here =====================
    # Instructions : Compute the cost of a particular choice of theta
    #                You should set cost and grad correctly.
    z = np.dot(X, theta)
    hyp = sigmoid(z)
    loss = hyp - y
    grad = np.dot(X.transpose(), loss)/m
    cost = -1*np.sum(y*np.log(hyp)+(1-y)*np.log(1-hyp))/m
    # ===========================================================

    return cost, grad
