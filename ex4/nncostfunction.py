import numpy as np
from sigmoid import *
from sigmoidgradient import *


def nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lmd):
    # Reshape nn_params back into the parameters theta1 and theta2, the weight 2-D arrays
    # for our two layer neural network
    theta1 = nn_params[:hidden_layer_size * (input_layer_size + 1)].reshape(hidden_layer_size, input_layer_size + 1)
    theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):].reshape(num_labels, hidden_layer_size + 1)

    # Useful value
    m = y.size
    # print(y)

    # You need to return the following variables correctly
    cost = 0
    theta1_grad = np.zeros(theta1.shape)  # 25 x 401
    theta2_grad = np.zeros(theta2.shape)  # 10 x 26

    # ===================== Your Code Here =====================
    # Instructions : You should complete the code by working thru the
    #                following parts
    #
    # Part 1 : Feedforward the neural network and return the cost in the
    #          variable cost. After implementing Part 1, you can verify that your
    #          cost function computation is correct by running ex4.py
    a1 = np.concatenate((np.ones((m, 1)), X), axis = 1)
    z2 = np.dot(a1, theta1.transpose())
    l2 = np.size(z2, 0)
    a2 = np.concatenate((np.ones((l2, 1)), sigmoid(z2)), axis = 1)
    z3 = np.dot(a2, theta2.transpose())
    a3 = sigmoid(z3)
    yy = np.zeros((m, num_labels))
    #下标：0代表1，... ，9代表10
    for i in range(m):
        yy[i][y[i]-1] = 1
    cost = np.sum(-yy*np.log(a3)-(1-yy)*np.log(1-a3))/m
    reg_cost = np.sum(np.power(theta1[:,1:],2))+np.sum(np.power(theta2[:,1:],2))
    cost += lmd*reg_cost/(2*m)

    # Part 2: Implement the backpropagation algorithm to compute the gradients
    #         theta1_grad and theta2_grad. You should return the partial derivatives of
    #         the cost function with respect to theta1 and theta2 in theta1_grad and
    #         theta2_grad, respectively. After implementing Part 2, you can check
    #         that your implementation is correct by running checkNNGradients
    #
    #         Note: The vector y passed into the function is a vector of labels
    #               containing values from 1..K. You need to map this vector into a 
    #               binary vector of 1's and 0's to be used with the neural network
    #               cost function.
    #
    #         Hint: We recommend implementing backpropagation using a for-loop
    #               over the training examples if you are implementing it for the 
    #               first time.

    # Part 3: Implement regularization with the cost function and gradients.
    #
    #         Hint: You can implement this around the code for
    #               backpropagation. That is, you can compute the gradients for
    #               the regularization separately and then add them to theta1_grad
    #               and theta2_grad from Part 2.
    delta3 = a3-yy
    delta2 = np.dot(delta3, theta2)*sigmoid_gradient(np.concatenate((np.ones((l2, 1)),z2), axis = 1))
    theta2_grad = np.dot(delta3.transpose(), a2)
    theta1_grad = np.dot(delta2[:,1:].transpose(), a1)
    theta2_grad = theta2_grad/m
    theta2_grad[:,1:] += lmd*theta2[:,1:]/m
    theta1_grad = theta1_grad/m
    theta1_grad[:,1:] += lmd*theta1[:,1:]/m



    # ====================================================================================
    # Unroll gradients
    grad = np.concatenate([theta1_grad.flatten(), theta2_grad.flatten()])

    return cost, grad
