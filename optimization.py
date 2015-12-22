#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import theano
import theano.tensor as T
from utils import shared_dataset, get_network
import matplotlib.pyplot as plt


def mini_batch_sgd_fancy(train_data, labels, xTrain_data, xTrain_labels,
                         learning_rate, L1_reg, L2_reg, epochs,
                         batch_size,
                         hidden_dim, model_type, model_file=None):
    """

    :param train_data: np array of training data, (n_examples x n_features)
    :param labels: np array of correct labels, (n_examples).
                   if correct label is 2, then the label is 2 not [0, 0, 1, ...]
    :param xTrain_data: same as train_data but for cross-training
    :param xTrain_labels: see labels
    :param learning_rate: learning rate used in parameter updating
    :param L1_reg: L1 regularization
    :param L2_reg: L2 regularization
    :param epochs: number of training epochs
    :param batch_size: split the training and cross-training data into batches this size
    :param hidden_dim: list of ints or int, dimensions of hidden layer in network
    :param model_type: Type
    :param model_file: Optional, file to load network from
    :return: trained model
    """
    # Preamble #
    # determine dimensionality of data and number of classes
    n_train_samples, data_dim = train_data.shape
    n_classes = len(set(labels))

    # compute number of mini-batches for training, validation and testing
    train_set_x, train_set_y = shared_dataset(train_data, labels, True)
    xtrain_set_x, xtrain_set_y = shared_dataset(xTrain_data, xTrain_labels, True)
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_xtrain_batches = xtrain_set_x.get_value(borrow=True).shape[0] / batch_size

    batch_index = T.lscalar()

    # containers to hold mini-batches
    x = T.matrix('x')
    y = T.ivector('y')

    net = get_network(x=x, in_dim=data_dim, n_classes=n_classes, hidden_dim=hidden_dim, type=model_type)

    if net is False:
        return False

    # cost function
    cost = (net.negative_log_likelihood(labels=y) + L1_reg * net.L1 + (L2_reg / n_train_samples) * net.L2_sq)

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

    if model_file is not None:
        net.load(model_file)

    # do the actual training
    best_xtrain_loss = np.inf
    batch_costs = []
    xtrain_accuracies = []
    xtrain_costs_bin = []

    for epoch in xrange(0, epochs):
        if epoch % 1000 == 0:
            # collect the costs on the cross-train data
            xtrain_costs = [xtrain_fcn(_) for _ in xrange(n_xtrain_batches)]
            avg_xtrain_cost = np.mean(xtrain_costs)
            avg_xtrain_accuracy = 100 * (1 - avg_xtrain_cost)

            # collect stuff for plotting
            xtrain_accuracies.append(avg_xtrain_accuracy)
            xtrain_costs_bin += xtrain_costs

            if epoch % 1000 == 0:
                print("At epoch {0}, accuracy {1}".format(epoch, avg_xtrain_accuracy))

            # if we're getting better, save the model
            if avg_xtrain_cost < best_xtrain_loss:
                net.write("./model{}.pkl".format(epoch))

        for i in xrange(n_train_batches):
            batch_avg_cost = train_fcn(i)
            if i % 1000:
                batch_costs.append(float(batch_avg_cost))

    #plt.plot(batch_costs)
    #plt.show()
    #plt.plot(xtrain_accuracies)
    #plt.show()
    #plt.plot(xtrain_costs_bin)
    return net
