import numpy as np


def estimate_gaussian(X):
    # Useful variables
    m, n = X.shape
    # print('shape of X: %d, %d' %(X.shape[0], X.shape[1]))
    print('shape of X: ', end='')
    print(X.shape)

    # You should return these values correctly
    mu = np.zeros(n)
    sigma2 = np.zeros(n)

    # ===================== Your Code Here =====================
    # Instructions: Compute the mean of the data and the variances
    #               In particular, mu[i] should contain the mean of
    #               the data for the i-th feature and sigma2[i]
    #               should contain variance of the i-th feature
    #
    mu = np.mean(X, 0)
    sigma2 = np.var(X, 0, ddof=0)
    #
    # mu = np.sum(X, 0)/m
    # sigma2 = np.sum(np.power(X-mu, 2), 0)/m

    print(mu)
    print(sigma2)


    # ==========================================================

    return mu, sigma2
