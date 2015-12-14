#!/usr/bin/env python
"""Code borrowed from:
http://cs231n.github.io/neural-networks-case-study/
http://nbviewer.ipython.org/github/dennybritz/nn-from-scratch/blob/master/nn-from-scratch.ipynb
"""

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib


def generate_2_class_moon_data():
    # seed random
    np.random.seed(0)

    # get some data
    X, y = sklearn.datasets.make_moons(300, noise=0.10)
    return X, y


def generate_3_class_spiral_data(plot=False):
    N = 100 # number of points per class
    D = 2 # dimensionality
    K = 3 # number of classes
    X = np.zeros((N*K,D)) # data matrix (each row = single example)
    y = np.zeros(N*K, dtype='uint8') # class labels
    for j in xrange(K):
      ix = range(N*j,N*(j+1))
      r = np.linspace(0.0,1,N) # radius
      t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
      X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
      y[ix] = j

    if plot is True:
        plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
        plt.show()

    return X, y


def plot_decision_boundary(pred_func, X, labels):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap=plt.cm.Spectral)
    plt.show()


def calculate_loss(model, input_data, labels, reg_lambda=0.01):
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


def predict(model, X):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2

    # output?
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)


def build_model(train_data, labels, nn_hdim, reg_lambda=0.01, epsilon=0.01, num_passes=10000, print_loss=False):
    # input dimensions is the size of the rows
    nn_input_dim = len(train_data[0, :])
    nn_output_dim = len(train_data[0, :])

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
            print "Loss after iteration %i: %f" %(i, calculate_loss(model=model, input_data=train_data,
                                                                    labels=labels))

    return model



X, y = generate_2_class_moon_data()
model = build_model(X, y, 3, print_loss=True)
plot_decision_boundary(lambda x: predict(model, x), X, y)
