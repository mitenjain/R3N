#!/usr/bin/env python

from neural_network import *

def hyperbolic_tangent(z, deriv=False):
    """The tanh function."""
    if deriv is True:
        return 1 - np.power(z, 2)
    else:
        return np.tanh(z)