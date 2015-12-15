#!/usr/bin/env python
"""
Citations
[1]http://cs231n.github.io/neural-networks-case-study/
[2]http://nbviewer.ipython.org/github/dennybritz/nn-from-scratch/blob/master/nn-from-scratch.ipynb
[3]https://github.com/mnielsen/neural-networks-and-deep-learning
"""

import numpy as np
import random


class NeuralNetwork(object):
    """[3], [2]
    """
    def __init__(self, dimensions):
        # eg. dimensions = [2, 10, 3] makes a 2-input, 10 hidden, 3 output NN
        self.layers = len(dimensions)
        np.random.seed(0)
        #self.weights = [np.random.randn(y, x)
        #                for x, y, in zip(dimensions[:-1], dimensions[1:])]
        #self.biases = [np.random.randn(y, 1) for y in dimensions]
        #W1 = 0.01 * np.random.randn(2, 50)  # 2=D, 50=h
        W1 = np.random.randn(2, 50) / np.sqrt(2)
        b1 = np.zeros((1, 50))

        W2 = np.random.randn(50, 3) / np.sqrt(50)
        b2 = np.zeros((1, 3))

        self.weights = [W1, W2]
        self.biases = [b1, b2]

    def feed_forward(self, a):
        for b, w in zip(self.biases, self.weights):
            z = np.dot(a, w) + b
            a = sigmoid(z)
        return a

    def stochastic_gradient_descent(self, train_data, labels, epsilon=0.01, num_passes=10000, print_loss=False):

        for i in xrange(num_passes):
            print "on iteration", i
            self.update(train_data, labels, epsilon)

    def update(self, in_data, labels, epsilon):
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]

        for sample, label in zip(in_data, labels):
            print sample, label
            delta_grad_b, delta_grad_w = self.backprop(sample, label)
            grad_b = [nb + dnb for nb, dnb in zip(grad_b, delta_grad_b)]
            grad_w = [nw + dnw for nw, dnw in zip(grad_w, delta_grad_w)]

        self.weights = [w - epsilon * nw for w, nw in zip(self.weights, grad_w)]
        self.biases = [b - epsilon * nb for b, nb in zip(self.biases, grad_b)]

    def fit(self, samples, labels, print_loss=False):
        for i in xrange(0, 10000):
            zs = []  # list to store z vectors
            activation = samples  # initialize to input data
            activations = [samples, ]  # list to store activations

            # forward pass
            for bias, weight in zip(self.biases, self.weights):
                z = np.dot(activation, weight) + bias
                zs.append(z)
                #activation = sigmoid(z_j)
                activation = np.tanh(z)
                activations.append(activation)

            exp_scores = np.exp(zs[-1])
            probs = exp_scores / np.sum(exp_scores)

            delta = self.cost_derivate2(probs, labels) * sigmoid_prime(zs[-1])

            # backward pass
            grad_b = [np.zeros(b.shape) for b in self.biases]
            grad_w = [np.zeros(w.shape) for w in self.weights]

            # initialize
            a_hidden = activations[-2]
            grad_w[-1] = np.dot(a_hidden.T, delta)
            grad_b[-1] = np.sum(delta, axis=0, keepdims=True)

            # loop backwards through the network
            for layer in xrange(2, self.layers):
                z = zs[-layer]
                #sp = sigmoid_prime(z)
                sp = 1 - np.power(activations[-layer], 2)
                delta = np.dot(delta, self.weights[-layer + 1].T) + sp
                grad_b[-layer] = delta
                grad_w[-layer] = np.dot(activations[-layer - 1].T, delta)

            epsilon = 0.01

            if print_loss and i % 1000 == 0:
                print "Loss after iteration %i: %f" % (i, self.calculate_loss(input_data=samples,
                                                                              labels=labels))

            self.weights = [w - (epsilon * nw) for w, nw in zip(self.weights, grad_w)]
            self.biases = [b - (epsilon * nb) for b, nb in zip(self.biases, grad_b)]

    def cost_derivate2(self, output_probs, labels):
        output_probs[range(len(labels)), labels] -= 1
        return output_probs

    def calculate_loss(self, input_data, labels, reg_lambda=0.01):
        num_examples = len(input_data)

        #W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

        W1 = self.weights[0]
        W2 = self.weights[1]
        b1 = self.biases[0]
        b2 = self.biases[1]

        # Forward propagation to calculate our predictions
        # layer 1
        z1 = input_data.dot(W1) + b1  # input to layer 1
        a1 = np.tanh(z1)  # output from layer 1

        # input to layer 2
        z2 = a1.dot(W2) + b2

        # output?
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # Calculating the loss
        corect_logprobs = -np.log(probs[range(num_examples), labels])
        data_loss = np.sum(corect_logprobs)

        # Add regulatization term to loss (optional)
        data_loss += reg_lambda / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
        return float(1. / num_examples * data_loss)


def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))