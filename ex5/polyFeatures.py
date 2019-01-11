import numpy as np


def poly_features(X, p):
    # You need to return the following variable correctly.
    X_poly = np.zeros((X.size, p))
    # print('size of X: '+str(X.shape))
    # ===================== Your Code Here =====================
    # Instructions : Given a vector X, return a matrix X_poly where the p-th
    #                column of X contains the values of X to the p-th power.
    for i in range(p):
    	X_poly[:,i] = np.power(X.transpose(), i+1)


    # ==========================================================

    return X_poly