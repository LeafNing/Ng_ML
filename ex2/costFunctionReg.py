import numpy as np
from sigmoid import *


def cost_function_reg(theta, X, y, lmd):
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
    cost = -1*np.sum(y*np.log(hyp)+(1-y)*np.log(1-hyp))/m+lmd*np.dot(theta.transpose(),theta)/(2*m)
    grad = np.dot(X.transpose(), loss)/m
    grad[1:] += lmd*theta[1:]/m

    # ===========================================================

    return cost, grad
