import matplotlib.pyplot as plt
import numpy as np

def plot_data(X, y):
    plt.figure()

    # ===================== Your Code Here =====================
    # Instructions : Plot the positive and negative examples on a
    #                2D plot, using the marker="+" for the positive
    #                examples and marker="o" for the negative examples
    label0 = np.where(y==0)
    label1 = np.where(y==1)

    plt.scatter(X[label0, 0], X[label0, 1], marker='o', color='yellow', label='negative')
    plt.scatter(X[label1, 0], X[label1, 1], marker='+', color='black', label='positive')
    plt.legend(['positive', 'negative'], loc=1)
    plt.show()

