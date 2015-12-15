#!/usr/bin/env python
"""
Citations
[1]http://cs231n.github.io/neural-networks-case-study/
[2]http://nbviewer.ipython.org/github/dennybritz/nn-from-scratch/blob/master/nn-from-scratch.ipynb
[3]https://github.com/mnielsen/neural-networks-and-deep-learning
"""

import numpy as np
import random

def calculate_loss2(model, input_data, labels, reg_lambda=0.01):
    num_examples = len(input_data)

    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

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


        nn_input_dim = 2
        nn_hdim = 50
        nn_output_dim = 3

        # W1 has shape (input_params, number_of_hidden_nodes)
        self.W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
        # b1 has shape (1, number_of_hidden_nodes)
        self.b1 = np.zeros((1, nn_hdim))

        # W2 takes the output from layer 1 as input so it has shape (nb_hidden_nodes, output_classes)
        self.W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
        self.b2 = np.zeros((1, nn_output_dim))

        self.weights = [self.W1, self.W2]
        self.biases = [self.b1, self.b2]

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
                z = activation.dot(weight) + bias
                zs.append(z)
                #activation = sigmoid(z_j)
                activation = np.tanh(z)
                activations.append(activation)

            #exp_scores = np.exp(zs[-2])
            exp_scores = np.exp(zs[-1])
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            #delta = self.cost_derivate2(probs, labels) * sigmoid_prime(zs[-1])
            delta = probs
            delta[range(len(samples)), labels] -= 1

            # backward pass

            # place to store gradients
            grad_w = [np.zeros(w.shape) for w in self.weights]
            grad_b = [np.zeros(b.shape) for b in self.biases]

            # initialize
            grad_w[-1] = (activations[-2].T).dot(delta)
            grad_b[-1] = np.sum(delta, axis=0, keepdims=True)

            # loop backwards through the network
            for layer in xrange(2, self.layers):
                z = zs[-layer]

                #sp = sigmoid_prime(z)
                #sp = 1 - np.power(activations[-layer], 2)
                #delta = delta.dot(self.weights[-layer + 1].T) * sp

                delta = delta.dot(self.weights[-layer + 1].T) * (1 - np.power(activations[-layer], 2))

                grad_w[-layer] = np.dot(activations[-layer - 1].T, delta)
                grad_b[-layer] = np.sum(delta, axis=0)

            epsilon = 0.01
            reg_lambda = 0.01

            grad_w[0] += reg_lambda * self.weights[0]
            grad_w[1] += reg_lambda * self.weights[1]

            self.weights[0] += -epsilon * grad_w[0]
            self.weights[1] += -epsilon * grad_w[1]
            self.biases[0] += -epsilon * grad_b[0]
            self.biases[1] += -epsilon * grad_b[1]

            #self.weights = [w - (epsilon * nw) for w, nw in zip(self.weights, grad_w)]
            #self.biases = [b - (epsilon * nb) for b, nb in zip(self.biases, grad_b)]

            if print_loss and i % 1000 == 0:
                W1 = self.weights[0]
                W2 = self.weights[1]
                b1 = self.biases[0]
                b2 = self.biases[1]
                model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
                print "Loss after iteration %i: %f" % (i, calculate_loss2(model=model, input_data=samples,
                                                                          labels=labels))
        a1 = np.tanh(samples.dot(self.weights[0]) + self.biases[0])
        scores = np.dot(a1, self.weights[1]) + self.biases[1]
        predict = np.argmax(scores, axis=1)
        print "training accuracy: %0.2f" % (np.mean(predict == labels))


    def build_model1(self, train_data, nb_classes, labels, nn_hdim, reg_lambda=0.01, epsilon=0.01, num_passes=10000,
                     print_loss=False):
        # input dimensions is the size of the rows
        nn_input_dim = len(train_data[0, :])

        nn_output_dim = nb_classes

        num_examples = len(train_data)
        assert(num_examples == len(labels))

        # Initialize the parameters to random values. We need to learn these.
        np.random.seed(0)

        # W1 has shape (input_params, number_of_hidden_nodes)
        W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
        # b1 has shape (1, number_of_hidden_nodes)
        b1 = np.zeros((1, nn_hdim))

        # W2 takes the output from layer 1 as input so it has shape (nb_hidden_nodes, output_classes)
        W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
        b2 = np.zeros((1, nn_output_dim))

        # This is what we return at the end
        model = {}

        # Gradient descent. For each batch...
        for i in xrange(0, num_passes):
            ## Forward propagation ##

            # input to layer 1 is X * W1 + b
            z1 = train_data.dot(W1) + b1

            # transform with activation function
            a1 = np.tanh(z1)

            # input to layer 2, take a1 * W2 + b2
            z2 = a1.dot(W2) + b2

            # get the scores
            exp_scores = np.exp(z2)

            # normalize
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            ## Backpropagation ##
            delta3 = probs

            # figure out how much we missed by. if the prob was 1 for the correct label then that
            # entry in delta3 will now be 0, otherwise it reflects how much error and in what direction
            delta3[range(num_examples), labels] -= 1

            dW2 = (a1.T).dot(delta3)
            db2 = np.sum(delta3, axis=0, keepdims=True)
            delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
            dW1 = np.dot(train_data.T, delta2)
            db1 = np.sum(delta2, axis=0)

            # Add regularization terms (b1 and b2 don't have regularization terms)
            dW2 += reg_lambda * W2
            dW1 += reg_lambda * W1

            # Gradient descent parameter update
            W1 += -epsilon * dW1
            b1 += -epsilon * db1
            W2 += -epsilon * dW2
            b2 += -epsilon * db2

            # Assign new parameters to the model
            model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 1000 == 0:
                print "Loss after iteration %i: %f" %(i, calculate_loss2(model=model, input_data=train_data,
                                                                         labels=labels))
        a1 = np.tanh(train_data.dot(W1) + b1)
        scores = np.dot(a1, W2) + b2
        predicted_class = np.argmax(scores, axis=1)
        print "training accuracy: %0.2f" % (np.mean(predicted_class == labels))

        return model

    def cost_derivate2(self, output_probs, labels):
        output_probs[range(len(labels)), labels] -= 1
        return output_probs

    def calculate_loss(self, input_data, labels, reg_lambda=0.01):
        num_examples = len(input_data)

        #W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

        #W1 = self.weights[0]
        #W2 = self.weights[1]
        #b1 = self.biases[0]
        #b2 = self.biases[1]

        W1 = self.W1
        W2 = self.W2
        b1 = self.b1
        b2 = self.b2

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