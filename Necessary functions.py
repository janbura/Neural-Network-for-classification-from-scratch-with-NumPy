import numpy as np


def initialize_parameters():
    W1 = np.random.uniform(-0.5, 0.5, (10, 784))
    b1 = np.random.uniform(-0.5, 0.5, (10, 1))
    W2 = np.random.uniform(-0.5, 0.5, (10, 10))
    b2 = np.random.uniform(-0.5, 0.5, (10, 1))
    return W1, b1, W2, b2


def ReLU(x):
    if x > 0:
        return x
    else:
        return 0

def softmax(x):
    return (np.exp(np.max(x))) / (np.sum(x))


def forward_prop(W1, b1, W2, b2, X):
    L1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    L2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return L1, A1, L2, A2