import numpy as np

def categorical_cross_entropy(y_true, y_pred):
    # y_true: one-hot vector (example: [0, 1, 0])
    # y_pred: propanility after softmax (example: [0.1, 0.8, 0.1])
    # add 1e-15, not get log(0)
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-15), axis=1))

def categorical_cross_entropy_prime(y_true, y_pred):
    return (y_pred - y_true) / y_true.shape[0]

