#!/usr/bin/env python
"""
Citations
[1]http://cs231n.github.io/neural-networks-case-study/
[2]http://nbviewer.ipython.org/github/dennybritz/nn-from-scratch/blob/master/nn-from-scratch.ipynb
[3]https://github.com/mnielsen/neural-networks-and-deep-learning
"""
from __future__ import print_function
from model import *
from utils import shared_dataset
import numpy as np
import sys
from itertools import izip
from layers import *
import cPickle
import theano
import theano.tensor as T


def predict(test_data, true_labels):
    x = T.matrix('x',)
    y = T.ivector('y')

    net = FastNeuralNetwork(x=x, in_dim=64, n_classes=10, hidden_dim=10)
    net.load("./model1500.pkl")
    print(net.input)

    predict_fcn = theano.function(inputs=[net.input],
                                  outputs=net.y_predict,
                                  )

    error_fcn = theano.function(inputs=[net.input, y],
                                outputs=net.errors(y),
                                )

    predictions = predict_fcn(test_data)
    errors = error_fcn(test_data, true_labels)

    print("prediction", predictions)
    print("errors", errors)
