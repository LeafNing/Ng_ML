import numpy as np


def linear_reg_cost_function(theta, x, y, lmd):
    # Initialize some useful values
    m = y.size

    # You need to return the following variables correctly
    cost = 0
    grad = np.zeros(theta.shape)

    # ===================== Your Code Here =====================
    # Instructions : Compute the cost and gradient of regularized linear
    #                regression for a particular choice of theta
    #
    #                You should set 'cost' to the cost and 'grad'
    #                to the gradient
    hyp = np.dot(x, theta)
    cost = 1/(2*m)*np.sum((hyp-y)**2)+lmd/(2*m)*np.sum(theta[1:]**2)
    grad = np.dot(x.transpose(), (hyp-y))/m
    grad[1:] += lmd/m*theta[1:]


    # ==========================================================

    return cost, grad
