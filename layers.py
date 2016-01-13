#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import theano
import theano.tensor as T

# globals
rng = np.random.RandomState()


class SoftmaxLayer(object):
    def __init__(self, x, in_dim, out_dim, id):
        self.weights = theano.shared(value=np.zeros([in_dim, out_dim], dtype=theano.config.floatX),
                                     name=id+'weights',
                                     borrow=True
                                     )
        self.biases = theano.shared(value=np.zeros([out_dim], dtype=theano.config.floatX),
                                    name=id+'biases',
                                    borrow=True
                                    )
        self.params = [self.weights, self.biases]

        self.input = x
        #self.output = ~T.isnan(self.prob_y_given_x(x)).any(axis=1)
        self.output = self.prob_y_given_x(x)
        # maybe put a switch here to check for nan/equivalent probs
        self.y_predict = T.argmax(self.output, axis=1)

    def prob_y_given_x(self, input_data):
        return T.nnet.softmax(T.dot(input_data, self.weights) + self.biases)

    def prob_y_given_x2(self, input_data):
        return T.switch(T.isnan(T.nnet.softmax(T.dot(input_data, self.weights) + self.biases)),
                        0, T.nnet.softmax(T.dot(input_data, self.weights) + self.biases))

    def negative_log_likelihood(self, labels):
        return -T.mean(T.log(self.output)[T.arange(labels.shape[0]), labels])

    def errors(self, labels):
        return T.mean(T.neq(self.y_predict, labels))


class HiddenLayer(object):
    def __init__(self, x, in_dim, out_dim, layer_id, W=None, b=None, activation=T.tanh):
        if W is None:
            W_values = np.asarray(rng.uniform(low=-np.sqrt(6. / (in_dim + out_dim)),
                                              high=np.sqrt(6. / (in_dim + out_dim)),
                                              size=(in_dim, out_dim)),
                                  dtype=theano.config.floatX
                                  )
            W = theano.shared(value=W_values, name=layer_id + 'weights', borrow=True)
        if b is None:
            b_values = np.zeros((out_dim,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name=layer_id + 'biases', borrow=True)

        self.weights = W
        self.biases = b
        self.params = [self.weights, self.biases]

        self.input = x
        lin_out = T.dot(x, self.weights) + self.biases
        self.output = lin_out if activation is None else activation(lin_out)
