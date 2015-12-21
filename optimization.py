#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import theano
import theano.tensor as T
from model import *
from neural_network import predict
from utils import shared_dataset
import matplotlib.pyplot as plt


def mini_batch_sgd_fancy(train_data, labels, xTrain_data, xTrain_labels,
                         learning_rate, L1_reg, L2_reg, epochs,
                         batch_size):
    # Preamble #
    # compute number of mini-batches for training, validation and testing
    train_set_x, train_set_y = shared_dataset(train_data, labels, True)
    xtrain_set_x, xtrain_set_y = shared_dataset(xTrain_data, xTrain_labels, True)
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_xtrain_batches = xtrain_set_x.get_value(borrow=True).shape[0] / batch_size

    batch_index = T.lscalar()

    # containers to hold mini-batches
    x = T.matrix('x')
    y = T.ivector('y')

    net = FastNeuralNetwork(x=x, in_dim=64, n_classes=10, hidden_dim=10)
    #net = ThreeLayerNetwork(x=x, in_dim=64, n_classes=10, hidden_dim=[30, 30])

    # cost function
    cost = (net.negative_log_likelihood(labels=y) + L1_reg * net.L1 + L2_reg * net.L2_sq)

    xtrain_fcn = theano.function(inputs=[batch_index],
                                 outputs=net.errors(y),
                                 givens={
                                     x: xtrain_set_x[batch_index * batch_size: (batch_index + 1) * batch_size],
                                     y: xtrain_set_y[batch_index * batch_size: (batch_index + 1) * batch_size]
                                 })

    # gradients
    nambla_params = [T.grad(cost, param) for param in net.params]

    # update tuple
    updates = [(param, param - learning_rate * nambla_param)
               for param, nambla_param in zip(net.params, nambla_params)]

    # main function? could make this an attribute and reduce redundant code
    train_fcn = theano.function(inputs=[batch_index],
                                outputs=cost,
                                updates=updates,
                                givens={
                                    x: train_set_x[batch_index * batch_size: (batch_index + 1) * batch_size],
                                    y: train_set_y[batch_index * batch_size: (batch_index + 1) * batch_size]
                                })

    # train the model
    best_xtrain_loss = np.inf
    batch_costs = []

    #net.load("./model1500.pkl")
    for epoch in xrange(0, epochs):
        if epoch % 500 == 0:
            xtrain_costs = [xtrain_fcn(_) for _ in xrange(n_xtrain_batches)]
            avg_xtrain_costs = np.mean(xtrain_costs)
            avg_xtrain_accuracy = 100 * (1 - avg_xtrain_costs)
            print("At epoch {0}, accuracy {1}".format(epoch, avg_xtrain_accuracy))
            # if we're getting better, save the model
            if avg_xtrain_costs < best_xtrain_loss:
                net.write("./model{}.pkl".format(epoch))
        for i in xrange(n_train_batches):
            batch_avg_cost = train_fcn(i)
            if i % 100:
                batch_costs.append(float(batch_avg_cost))
    plt.plot(batch_costs)
    plt.show()


def mini_batch_sgd(train_data, labels, xTrain_data, xTrain_labels,
                   learning_rate, L1_reg, L2_reg, epochs,
                   batch_size):
    # compute number of minibatches for training, validation and testing
    train_set_x, train_set_y = shared_dataset(train_data, labels, True)
    valid_set_x, valid_set_y = shared_dataset(xTrain_data, xTrain_labels, True)
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size

    batch_index = T.lscalar()

    # containers to hold mini-batches
    x = T.matrix('x')
    y = T.ivector('y')

    net = FastNeuralNetwork(x=x, in_dim=64, n_classes=10, hidden_dim=10)
    #net = ThreeLayerNetwork(x=x, in_dim=64, n_classes=10, hidden_dim=[10, 10])

    # cost function
    cost = (net.negative_log_likelihood(labels=y) + L1_reg * net.L1 + L2_reg * net.L2_sq)

    valid_fcn = theano.function(inputs=[batch_index],
                                outputs=net.errors(y),
                                givens={
                                    x: valid_set_x[batch_index * batch_size: (batch_index + 1) * batch_size],
                                    y: valid_set_y[batch_index * batch_size: (batch_index + 1) * batch_size]
                                })
    print(net.params)
    # gradients
    nambla_params = [T.grad(cost, param) for param in net.params]

    # update tuple
    updates = [(param, param - learning_rate * nambla_param)
               for param, nambla_param in zip(net.params, nambla_params)]

    # main function? could make this an attribute and reduce redundant code
    train_fcn = theano.function(inputs=[batch_index],
                                outputs=cost,
                                updates=updates,
                                givens={
                                    x: train_set_x[batch_index * batch_size: (batch_index + 1) * batch_size],
                                    y: train_set_y[batch_index * batch_size: (batch_index + 1) * batch_size]
                                })

    # train the model
    net.load("./model.pkl")
    for epoch in xrange(0, epochs):
        if epoch % 500 == 0:
            valid_costs = [valid_fcn(_) for _ in xrange(n_valid_batches)]
            mean_validation_cost = 100 * (1 - np.mean(valid_costs))
            print("At epoch {0}, accuracy {1}".format(epoch, mean_validation_cost))
        for i in xrange(n_train_batches):
            batch_avg_cost = train_fcn(i)
    net.write("./model.pkl")
    return net
