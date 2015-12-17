#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib
from random import shuffle

def load_iris_dataset():
    data = sklearn.datasets.load_iris()
    X = data.data
    y = data.target
    return X, y


def load_digit_dataset(dataset_size, split):
    data = sklearn.datasets.load_digits()
    X = data.data
    y = data.target
    whole_dataset = zip(X, y)
    shuffle(whole_dataset)
    whole_dataset = whole_dataset[:dataset_size]
    train_dataset = whole_dataset[:int(dataset_size * (1 - split))]
    test_dataset = whole_dataset[int(dataset_size * (1 - split)):]

    return train_dataset, test_dataset


def generate_2_class_moon_data(size=300, noise=0.1):
    # seed random
    np.random.seed(0)

    # get some data
    X, y = sklearn.datasets.make_moons(size, noise=noise)
    return X, y


def generate_3_class_spiral_data(nb_classes, theta=0.2, plot=False):
    N = 200 # number of points per class
    D = 2 # dimensionality
    K = nb_classes  # number of classes
    X = np.zeros((N*K,D)) # data matrix (each row = single example)
    y = np.zeros(N*K, dtype='uint8') # class labels
    for j in xrange(K):
      ix = range(N*j,N*(j+1))
      r = np.linspace(0.0,1,N) # radius
      t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N) * theta # theta
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