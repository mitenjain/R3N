#!/usr/bin/env python
from __future__ import print_function
import os, sys
import cPickle
import theano
import theano.tensor as T
import numpy as np
from utils import shared_dataset, get_network


def mini_batch_sgd(motif, train_data, labels, xTrain_data, xTrain_targets,
                   learning_rate, L1_reg, L2_reg, epochs,
                   batch_size,
                   hidden_dim, model_type, model_file=None,
                   trained_model_dir=None, verbose=True, extra_args=None
                   ):
    # Preamble #
    # determine dimensionality of data and number of classes
    n_train_samples, data_dim = train_data.shape
    n_classes = len(set(labels))

    # compute number of mini-batches for training, validation and testing
    train_set_x, train_set_y = shared_dataset(train_data, labels, True)
    xtrain_set_x, xtrain_set_y = shared_dataset(xTrain_data, xTrain_targets, True)
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_xtrain_batches = xtrain_set_x.get_value(borrow=True).shape[0] / batch_size

    batch_index = T.lscalar()

    # containers to hold mini-batches
    x = T.matrix('x')
    y = T.ivector('y')

    net = get_network(x=x, in_dim=data_dim, n_classes=n_classes, hidden_dim=hidden_dim, model_type=model_type,
                      extra_args=extra_args)

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
    batch_costs = [np.inf]
    xtrain_accuracies = []
    xtrain_costs_bin = []

    check_frequency = int(epochs / 10)

    for epoch in xrange(0, epochs):
        if epoch % check_frequency == 0:
            # collect the costs on the cross-train data
            xtrain_costs = [xtrain_fcn(_) for _ in xrange(n_xtrain_batches)]
            avg_xtrain_cost = np.mean(xtrain_costs)
            avg_xtrain_accuracy = 100 * (1 - avg_xtrain_cost)

            # collect stuff for plotting
            xtrain_accuracies.append(avg_xtrain_accuracy)
            xtrain_costs_bin += xtrain_costs

            if verbose:
                print("{0}: epoch {1}, batch cost {2}, cross-train accuracy {3}".format(motif, epoch, batch_costs[-1],
                                                                                        avg_xtrain_accuracy),
                      file=sys.stderr)

            # if we're getting better, save the model
            if avg_xtrain_cost < best_xtrain_loss and trained_model_dir is not None:
                if not os.path.exists(trained_model_dir):
                    os.makedirs(trained_model_dir)
                net.write("{0}model{1}.pkl".format(trained_model_dir, epoch))

        for i in xrange(n_train_batches):
            batch_avg_cost = train_fcn(i)
            try:
                if i % (n_train_batches / 10) == 0:
                    batch_costs.append(float(batch_avg_cost))
            except ZeroDivisionError:
                pass

    # pickle the summary stats for the training
    summary = {
        "batch_costs": batch_costs,
        "xtrain_accuracies": xtrain_accuracies,
        "xtrain_costs": xtrain_costs_bin
    }
    if trained_model_dir is not None:
        with open("{}summary_stats.pkl".format(trained_model_dir), 'w') as f:
            cPickle.dump(summary, f)

    return net, summary


def mini_batch_sgd_with_annealing(motif, train_data, labels, xTrain_data, xTrain_targets,
                                  learning_rate, L1_reg, L2_reg, epochs,
                                  batch_size,
                                  hidden_dim, model_type, model_file=None,
                                  trained_model_dir=None, verbose=False, extra_args=None):
    # Preamble #
    # determine dimensionality of data and number of classes
    n_train_samples, data_dim = train_data.shape
    n_classes = len(set(labels))

    # compute number of mini-batches for training, validation and testing
    train_set_x, train_set_y = shared_dataset(train_data, labels, True)
    xtrain_set_x, xtrain_set_y = shared_dataset(xTrain_data, xTrain_targets, True)
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_xtrain_batches = xtrain_set_x.get_value(borrow=True).shape[0] / batch_size

    batch_index = T.lscalar()

    # containers to hold mini-batches
    x = T.matrix('x')
    y = T.ivector('y')

    net = get_network(x=x, in_dim=data_dim, n_classes=n_classes, hidden_dim=hidden_dim, model_type=model_type,
                      extra_args=extra_args)

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
    dynamic_learning_rate = T.as_tensor_variable(learning_rate)

    #dynamic_learning_rate = learning_rate
    updates = [(param, param - dynamic_learning_rate * nambla_param)
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
    batch_costs = [np.inf]
    xtrain_accuracies = []
    xtrain_costs_bin = []
    prev_xtrain_cost = 1e-10
    check_frequency = epochs / 10

    for epoch in xrange(0, epochs):
        # evaluation of training progress and summary stat collection
        if epoch % check_frequency == 0:
            # collect the costs on the cross-train data
            xtrain_errors = [xtrain_fcn(_) for _ in xrange(n_xtrain_batches)]
            avg_xtrain_errors = np.mean(xtrain_errors)
            avg_xtrain_accuracy = 100 * (1 - avg_xtrain_errors)

            # collect stuff for plotting
            xtrain_accuracies.append(avg_xtrain_accuracy)
            xtrain_costs_bin += xtrain_errors

            if verbose:
                print("{0}: epoch {1}, batch cost {2}, cross-train accuracy {3}".format(motif, epoch,
                                                                                        batch_costs[-1],
                                                                                        avg_xtrain_accuracy),
                      file=sys.stderr)

            # if we're getting better, save the model
            if avg_xtrain_errors < best_xtrain_loss and trained_model_dir is not None:
                if not os.path.exists(trained_model_dir):
                    os.makedirs(trained_model_dir)
                net.write("{0}model{1}.pkl".format(trained_model_dir, epoch))

        for i in xrange(n_train_batches):
            batch_avg_cost = train_fcn(i)
            if i % (n_train_batches / 10) == 0:
                batch_costs.append(float(batch_avg_cost))

        # annealing protocol
        mean_xtrain_cost = np.mean([xtrain_fcn(_) for _ in xrange(n_xtrain_batches)])
        if mean_xtrain_cost / prev_xtrain_cost < 1.0:
            dynamic_learning_rate *= 0.9

        if mean_xtrain_cost > prev_xtrain_cost:
            #print("GOT WORSE")
            dynamic_learning_rate *= 1.05
        prev_xtrain_cost = mean_xtrain_cost

    # pickle the summary stats for the training
    summary = {
        "batch_costs": batch_costs,
        "xtrain_accuracies": xtrain_accuracies,
        "xtrain_errors": xtrain_costs_bin
    }
    if trained_model_dir is not None:
        with open("{}summary_stats.pkl".format(trained_model_dir), 'w') as f:
            cPickle.dump(summary, f)

    return net, summary
