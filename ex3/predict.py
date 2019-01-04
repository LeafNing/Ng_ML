import numpy as np
from sigmoid import *

def predict(theta1, theta2, x):
    # Useful values
    m = x.shape[0]
    num_labels = theta2.shape[0]

    # You need to return the following variable correctly
    p = np.zeros(m)
    x = np.c_[np.ones(m), x]

    # ===================== Your Code Here =====================
    # Instructions : Complete the following code to make predictions using
    #                your learned neural network. You should set p to a
    #                1-D array containing labels between 1 to num_labels.
    tmp0 = sigmoid(np.dot(x, theta1.transpose()))
    tmp1 = np.c_[np.ones(m), tmp0]
    tmp2 = sigmoid(np.dot(tmp1, theta2.transpose()))
    p = np.argmax(tmp2, axis=1)+1


    return p


