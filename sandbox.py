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


def generate_3_class_spiral_data(nb_classes, plot=False):
    N = 100 # number of points per class
    D = 2 # dimensionality
    K = nb_classes  # number of classes
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
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])

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


def plot_decision_boundry2():
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = np.dot(np.maximum(0, np.dot(np.c_[xx.ravel(), yy.ravel()], W) + b), W2) + b2
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)
    fig = plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
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


def build_model1(train_data, nb_classes, labels, nn_hdim, reg_lambda=0.01, epsilon=0.01, num_passes=10000,
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
            print "Loss after iteration %i: %f" %(i, calculate_loss(model=model, input_data=train_data,
                                                                    labels=labels))
    a1 = np.tanh(train_data.dot(W1) + b1)
    scores = np.dot(a1, W2) + b2
    predicted_class = np.argmax(scores, axis=1)
    print "training accuracy: %0.2f" % (np.mean(predicted_class == labels))

    return model


def build_model2(input_data, labels, nb_classes, nn_hidden_dim):
    # initialize parameters randomly
    X = input_data
    h = nn_hidden_dim  # size of hidden layer
    D = input_data.shape[1]
    W = 0.01 * np.random.randn(D, h)
    b = np.zeros((1, h))

    K = nb_classes  # similar to nn_output_dim
    W2 = 0.01 * np.random.randn(h, K)
    b2 = np.zeros((1, K))

    # some hyperparameters
    step_size = 1e-0
    reg = 1e-3  # regularization strength

    # gradient descent loop
    num_examples = X.shape[0]

    model = {}

    for i in xrange(10000):
        # evaluate class scores, [N x K]
        # this could be forward prop
        hidden_layer = np.maximum(0, np.dot(X, W) + b) # note, ReLU activation
        scores = np.dot(hidden_layer, W2) + b2

        # compute the class probabilities
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]

        # compute the loss: average cross-entropy loss and regularization
        # make this it's own function
        corect_logprobs = -np.log(probs[range(num_examples), labels])
        data_loss = np.sum(corect_logprobs) / num_examples
        reg_loss = 0.5 * reg * np.sum(W * W) + 0.5 * reg * np.sum(W2 * W2)
        loss = data_loss + reg_loss
        if i % 1000 == 0:
            print "iteration %d: loss %f" % (i, loss)

        # compute the gradient on scores
        dscores = probs
        dscores[range(num_examples), y] -= 1
        dscores /= num_examples

        # backpropate the gradient to the parameters
        # first backprop into parameters W2 and b2
        dW2 = np.dot(hidden_layer.T, dscores)
        db2 = np.sum(dscores, axis=0, keepdims=True)

        # next backprop into hidden layer
        dhidden = np.dot(dscores, W2.T)

        # backprop the ReLU non-linearity
        dhidden[hidden_layer <= 0] = 0

        # finally into W,b
        dW = np.dot(X.T, dhidden)
        db = np.sum(dhidden, axis=0, keepdims=True)

        # add regularization gradient contribution
        dW2 += reg * W2
        dW += reg * W

        # perform a parameter update
        W += -step_size * dW
        b += -step_size * db
        W2 += -step_size * dW2
        b2 += -step_size * db2

        model = {'W1': W, 'b1': b, 'W2': W2, 'b2': b2}

    hidden_layer = np.maximum(0, np.dot(X, W) + b)
    scores = np.dot(hidden_layer, W2) + b2
    predicted_class = np.argmax(scores, axis=1)
    print "training accuracy: %0.2f" % (np.mean(predicted_class == labels))


    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = np.dot(np.maximum(0, np.dot(np.c_[xx.ravel(), yy.ravel()], W) + b), W2) + b2
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)
    fig = plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()

    return model


X, y = generate_3_class_spiral_data(nb_classes=3, plot=False)

#m = build_model2(X, y, 3, 50)
m = build_model1(train_data=X, labels=y, nb_classes=3, nn_hdim=50, print_loss=True)
plot_decision_boundary(lambda x: predict(m, x), X, y)





