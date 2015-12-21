#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import theano
import theano.tensor as T
from neural_network import *
from utils import shared_dataset


def mini_batch_sgd(train_data, labels, valid_data, valid_labels,
                   learning_rate, L1_reg, L2_reg, epochs,
                   batch_size):
    # compute number of minibatches for training, validation and testing
    train_set_x, train_set_y = shared_dataset(train_data, labels, True)
    valid_set_x, valid_set_y = shared_dataset(valid_data, valid_labels, True)
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
