import numpy as np
import random

def kmeans_init_centroids(X, K):
    # You should return this value correctly
    centroids = np.zeros((K, X.shape[1]))

    # ===================== Your Code Here =====================
    # Instructions: You should set centroids to randomly chosen examples from
    #               the dataset X
    randidx = np.random.permutation(X.shape[0])
    centroids = X[randidx[0:K]]


    # ==========================================================

    return centroids
