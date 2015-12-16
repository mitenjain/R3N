#!/usr/bin/env python

from neural_network import *

def hyperbolic_tangent(z, deriv=False):
    """The tanh function."""
    if deriv is True:
        return 1 - np.power(z, 2)
    else:
        return np.tanh(z)


def ReLU(z, deriv=False):
    return np.maximum(0, z)


def soft_plus(z, deriv=False):
    if deriv is True:
        return 1.0 / (1 + np.exp(-z))
    else:
        return np.log(1 + np.exp(z))