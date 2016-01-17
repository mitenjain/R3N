#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

# globals
rng = np.random.RandomState()


class SoftmaxLayer(object):
    def __init__(self, x, in_dim, out_dim, layer_id):
        self.weights = theano.shared(value=np.zeros([in_dim, out_dim], dtype=theano.config.floatX),
                                     name=layer_id + 'weights',
                                     borrow=True
                                     )
        self.biases = theano.shared(value=np.zeros([out_dim], dtype=theano.config.floatX),
                                    name=layer_id + 'biases',
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


class ConvPoolLayer(object):
    def __init__(self, x, filter_shape, image_shape, poolsize, layer_id):
        #assert(image_shape[1] == filter_shape[1])

        self.input = x

        fan_in = np.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) / np.prod(poolsize))

        W_bound = np.sqrt(6. / (fan_in + fan_out))

        self.weights = theano.shared(
            np.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.biases = theano.shared(value=b_values, borrow=True)

        conv_out = conv.conv2d(
            input=x,
            filters=self.weights,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        self.output = T.tanh(pooled_out + self.biases.dimshuffle('x', 0, 'x', 'x'))

        self.params = [self.weights, self.biases]

