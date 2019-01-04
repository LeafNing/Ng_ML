import numpy as np
from sigmoid import *


def lr_cost_function(theta, X, y, lmd):
    m = y.size

    # You need to return the following values correctly
    cost = 0
    

    # ===================== Your Code Here =====================
    # Instructions : Compute the cost of a particular choice of theta
    #                You should set cost and grad correctly.
    z = np.dot(X, theta)
    hyp = sigmoid(z)
    loss = hyp - y 
    # theta1 = theta[1:]
    cost = -1*np.sum(y*np.log(hyp)+(1-y)*np.log(1-hyp))/m+lmd*np.dot(theta[1:].transpose(),theta[1:])/(2*m)
    


    # =========================================================

    return cost

def lr_grad_function(theta, X, y, lmd):
    m = y.size
    grad = np.zeros(theta.shape)
    z = np.dot(X, theta)
    hyp = sigmoid(z)
    loss = hyp - y 
    grad = np.dot(X.transpose(), loss)/m
    grad[1:] += lmd*theta[1:]/m
    return grad
