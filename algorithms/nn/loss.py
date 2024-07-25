import numpy as np


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true)

def cross_entropy(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred))

def cross_entropy_derivative(y_true, y_pred):
    return y_pred - y_true

def binary_cross_entropy(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_derivative(y_true, y_pred):
    for i in range(len(y_pred)):
        if y_pred[i] == 0:
            raise ValueError(f'y_pred cannot be 0, {y_pred[i]}')
        elif y_pred[i] == 1:
            raise ValueError(f'y_pred cannot be 1, {y_pred[i]}')
    return (y_pred - y_true) / (y_pred * (1 - y_pred))