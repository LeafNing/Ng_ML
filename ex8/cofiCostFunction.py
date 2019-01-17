import numpy as np


def cofi_cost_function(params, Y, R, num_users, num_movies, num_features, lmd):
    X = params[0:num_movies * num_features].reshape((num_movies, num_features))
    theta = params[num_movies * num_features:].reshape((num_users, num_features))

    # You need to set the following values correctly.
    cost = 0
    X_grad = np.zeros(X.shape)
    theta_grad = np.zeros(theta.shape)

    # ===================== Your Code Here =====================
    # Instructions: Compute the cost function and gradient for collaborative
    #               filtering. Concretely, you should first implement the cost
    #               function (without regularization) and make sure it is
    #               matches our costs. After that, you should implement the
    #               gradient and use the checkCostFunction routine to check
    #               that the gradient is correct. Finally, you should implement
    #               regularization.
    #
    # Notes: X - num_movies x num_features matrix of movie features
    #        theta - num_users x num_features matrix of user features
    #        Y - num_movies x num_users matrix of user ratings of movies
    #        R - num_movies x num_users matrix, where R[i, j] = 1 if the
    #        i-th movie was rated by the j-th user
    #
    # You should set the following variables correctly
    #
    #        X_grad - num_movies x num_features matrix, containing the
    #                 partial derivatives w.r.t. to each element of X
    #        theta_grad - num_users x num_features matrix, containing the
    #                     partial derivatives w.r.t. to each element of theta
    hyp = np.dot(X, theta.transpose())
    cost = 0.5*np.sum(R*(hyp-Y)**2)+0.5*lmd*np.sum(theta**2)+0.5*lmd*np.sum(X**2)

    X_grad = np.dot(R*(hyp-Y), theta)+lmd*X
    theta_grad = np.dot((R*(hyp-Y)).transpose(), X)+lmd*theta

    # for i in range(np.size(X, 0)):
    #     idx = R[i,:]==1
    #     X_grad[i,:] = (np.dot(X[i,:], theta[idx,:].transpose())-Y[i,idx]).dot(theta[idx,:])+lmd*X[i,:]
    # for j in range(np.size(theta, 0)):
    #     jdx = R[:,j]==1
    #     theta_grad[j,:] = (theta[j,:].dot(X[jdx,:].transpose())-Y[jdx,j]).dot(X[jdx,:])+lmd*theta[j,:]

    # ==========================================================

    grad = np.concatenate((X_grad.flatten(), theta_grad.flatten()))

    return cost, grad
