#!/usr/bin/env python
"""
Citations
[1]http://cs231n.github.io/neural-networks-case-study/
[2]http://nbviewer.ipython.org/github/dennybritz/nn-from-scratch/blob/master/nn-from-scratch.ipynb
[3]https://github.com/mnielsen/neural-networks-and-deep-learning
"""

import numpy as np


class NeuralNetwork(object):
    """[3], [2]
    """
    def __init__(self, dimensions):
        # eg. dimensions = [2, 10, 3] makes a 2-input, 10 hidden, 3 output NN
        self.layers = len(dimensions)
        np.random.seed(0)
        self.weights = [np.random.randn(y, x)
                        for x, y, in zip(dimensions[:-1], dimensions[1:])]
        self.biases = [np.random.randn(y, 1) for y in dimensions]

    def feed_forward(self, a):
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a) + b
            a = sigmoid(z)
        return a

    def stochastic_gradient_descent(self, train_data, ):



def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))