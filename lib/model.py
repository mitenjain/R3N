#!/usr/bin/env python
"""
Citations
[1]http://cs231n.github.io/neural-networks-case-study/
[2]http://nbviewer.ipython.org/github/dennybritz/nn-from-scratch/blob/master/nn-from-scratch.ipynb
[3]https://github.com/mnielsen/neural-networks-and-deep-learning
"""
from __future__ import print_function
import numpy as np
import sys
from itertools import izip
from layers import HiddenLayer, SoftmaxLayer, ConvPoolLayer
import cPickle
import theano.tensor as T


def ReLU(x):
        return T.switch(x < 0, 0, x)


class Model(object):
    """Base class for network models
    """
    def __init__(self, x, in_dim, n_classes, hidden_dim):
        """x: input data
          in_dim: dimensionality of input data
          n_classes: number of classes within data
        """
        self.input = x
        self.in_dim = in_dim
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.params = None
        self.initialized = False

    def write(self, file_path):
        """Write model to file, using cPickle
        file_path: string, path to and including file to be written
        """
        f = open(file_path, 'w')
        d = {
            "model": self.__class__,
            "in_dim": self.in_dim,
            "n_classes": self.n_classes,
            "hidden_dim": self.hidden_dim,
        }
        assert (self.params is not None)
        for param in self.params:
            lb = '{}'.format(param)
            d[lb] = param.get_value()
        cPickle.dump(d, f)

    def load_from_file(self, file_path=None, model_obj=None, careful=False):
        """Load model
         file_path: string, file to and including model file
        """
        if file_path is not None:
            f = open(file_path, 'r')
            d = cPickle.load(f)
        else:
            assert(model_obj is not None), "need to provide file or dict with model params"
            d=model_obj

        assert(self.in_dim == d['in_dim'])
        assert(self.n_classes == d['n_classes']), "Incorrect number of input classes, got {0} should be {1}".format(
            d['n_classes'], self.n_classes)
        assert(self.__class__ == d['model'])
        assert(self.hidden_dim == d['hidden_dim'])

        missing_params = 0
        for param in self.params:
            look_up = "{}".format(param)
            if look_up in d.keys():
                assert(len(param.get_value()) == len(d[look_up]))
                param.set_value(d[look_up])
            else:
                missing_params += 1
        if careful is True:
            print("got {} missing params".format(missing_params), file=sys.stderr)

    def load_from_object(self, model, careful=False):
        self.load_from_file(file_path=None, model_obj=model, careful=careful)


class NeuralNetwork(Model):
    def __init__(self, x, in_dim, hidden_dim, n_classes):
        assert(len(hidden_dim) == 1)
        super(NeuralNetwork, self).__init__(x=x, in_dim=in_dim, n_classes=n_classes, hidden_dim=hidden_dim)

        self.input = x

        # first layer (hidden)
        self.hidden_layer0 = HiddenLayer(x=x, in_dim=in_dim, out_dim=hidden_dim[0], layer_id='h1', activation=T.tanh)

        # final layer (softmax)
        self.softmax_layer = SoftmaxLayer(x=self.hidden_layer0.output, in_dim=hidden_dim[0], out_dim=n_classes,
                                          layer_id='s0')

        # Regularization
        self.L1 = abs(self.hidden_layer0.weights).sum() + abs(self.softmax_layer.weights).sum()
        self.L2_sq = (self.hidden_layer0.weights ** 2).sum() + (self.softmax_layer.weights ** 2).sum()

        # output, errors, and likelihood
        self.y_predict = self.softmax_layer.y_predict
        self.negative_log_likelihood = self.softmax_layer.negative_log_likelihood
        self.errors = self.softmax_layer.errors
        self.output = self.softmax_layer.output
        self.params = self.hidden_layer0.params + self.softmax_layer.params


        # network type
        self.type = "twoLayer"


class ThreeLayerNetwork(Model):
    def __init__(self, x, in_dim, hidden_dim, n_classes):
        assert(len(hidden_dim) == 2)
        # Note: hidden dim is a list with the dimensions of the hidden layer
        super(ThreeLayerNetwork, self).__init__(x=x, in_dim=in_dim, n_classes=n_classes, hidden_dim=hidden_dim)

        self.input = x

        # first layer (hidden)
        self.hidden_layer0 = HiddenLayer(x=x, in_dim=in_dim, out_dim=hidden_dim[0], activation=T.tanh, layer_id='h0')

        # second layer
        self.hidden_layer1 = HiddenLayer(x=self.hidden_layer0.output, in_dim=hidden_dim[0], out_dim=hidden_dim[1],
                                         activation=T.tanh, layer_id='h1')

        # final layer (softmax)
        self.softmax_layer = SoftmaxLayer(x=self.hidden_layer1.output, in_dim=hidden_dim[1], out_dim=n_classes,
                                          layer_id='s0')

        # Regularization
        self.L1 = abs(self.hidden_layer0.weights).sum() + abs(self.hidden_layer1.weights).sum() + \
                  abs(self.softmax_layer.weights).sum()
        self.L2_sq = (self.hidden_layer0.weights ** 2).sum() + abs(self.hidden_layer1.weights ** 2).sum() + \
                     (self.softmax_layer.weights ** 2).sum()

        # output, errors, and likelihood
        self.y_predict = self.softmax_layer.y_predict
        self.negative_log_likelihood = self.softmax_layer.negative_log_likelihood
        self.errors = self.softmax_layer.errors
        self.output = self.softmax_layer.output
        self.params = self.hidden_layer0.params + self.hidden_layer1.params + self.softmax_layer.params
        # type
        self.type = "threeLayer"


class ReLUThreeLayerNetwork(Model):
    def __init__(self, x, in_dim, hidden_dim, n_classes):
        assert(len(hidden_dim) == 2)
        # Note: hidden dim is a list with the dimensions of the hidden layer
        super(ReLUThreeLayerNetwork, self).__init__(x=x, in_dim=in_dim, n_classes=n_classes, hidden_dim=hidden_dim)

        self.input = x

        # first layer (hidden)
        self.hidden_layer0 = HiddenLayer(x=x, in_dim=in_dim, out_dim=hidden_dim[0], activation=ReLU,
                                         layer_id='h0')

        # second layer
        self.hidden_layer1 = HiddenLayer(x=self.hidden_layer0.output, in_dim=hidden_dim[0], out_dim=hidden_dim[1],
                                         activation=T.tanh, layer_id='h1')

        # final layer (softmax)
        self.softmax_layer = SoftmaxLayer(x=self.hidden_layer1.output, in_dim=hidden_dim[1], out_dim=n_classes,
                                          layer_id='s0')

        # Regularization
        self.L1 = abs(self.hidden_layer0.weights).sum() + abs(self.hidden_layer1.weights).sum() + \
                  abs(self.softmax_layer.weights).sum()
        self.L2_sq = (self.hidden_layer0.weights ** 2).sum() + abs(self.hidden_layer1.weights ** 2).sum() + \
                     (self.softmax_layer.weights ** 2).sum()

        # output, errors, and likelihood
        self.y_predict = self.softmax_layer.y_predict
        self.negative_log_likelihood = self.softmax_layer.negative_log_likelihood
        self.output = self.softmax_layer.output
        self.errors = self.softmax_layer.errors
        self.params = self.hidden_layer0.params + self.hidden_layer1.params + self.softmax_layer.params

        self.type = "ReLUthreeLayer"


class FourLayerNetwork(Model):
    def __init__(self, x, in_dim, hidden_dim, n_classes):
        assert(len(hidden_dim) == 3)
        super(FourLayerNetwork, self).__init__(x=x, in_dim=in_dim, n_classes=n_classes, hidden_dim=hidden_dim)

        # first layer (hidden)
        self.hidden_layer0 = HiddenLayer(x=x, in_dim=in_dim, out_dim=hidden_dim[0], activation=T.tanh, layer_id='h0')

        # second layer
        self.hidden_layer1 = HiddenLayer(x=self.hidden_layer0.output, in_dim=hidden_dim[0], out_dim=hidden_dim[1],
                                         activation=T.tanh, layer_id='h1')
        # third layer
        self.hidden_layer2 = HiddenLayer(x=self.hidden_layer1.output, in_dim=hidden_dim[1], out_dim=hidden_dim[2],
                                         activation=T.tanh, layer_id='h2')

        # final layer (softmax)
        self.softmax_layer = SoftmaxLayer(x=self.hidden_layer2.output, in_dim=hidden_dim[2], out_dim=n_classes,
                                          layer_id='s0')

        # Regularization
        self.L1 = abs(self.hidden_layer0.weights).sum() + abs(self.hidden_layer1.weights).sum() + \
                  abs(self.hidden_layer2.weights).sum() + abs(self.softmax_layer.weights).sum()
        self.L2_sq = (self.hidden_layer0.weights ** 2).sum() + abs(self.hidden_layer1.weights ** 2).sum() + \
                     (self.hidden_layer2.weights ** 2).sum() + (self.softmax_layer.weights ** 2).sum()

        # output, errors, and likelihood
        self.y_predict = self.softmax_layer.y_predict
        self.negative_log_likelihood = self.softmax_layer.negative_log_likelihood
        self.errors = self.softmax_layer.errors
        self.output = self.softmax_layer.output
        self.params = self.hidden_layer0.params + self.hidden_layer1.params + self.hidden_layer2.params + \
                      self.softmax_layer.params
        self.input = x
        self.type = "fourLayer"


class FourLayerReLUNetwork(Model):
    def __init__(self, x, in_dim, hidden_dim, n_classes):
        assert(len(hidden_dim) == 3)
        super(FourLayerReLUNetwork, self).__init__(x=x, in_dim=in_dim, n_classes=n_classes, hidden_dim=hidden_dim)

        self.input = x

        # first layer (hidden)
        self.hidden_layer0 = HiddenLayer(x=x, in_dim=in_dim, out_dim=hidden_dim[0], activation=ReLU, layer_id='h0')

        # second layer
        self.hidden_layer1 = HiddenLayer(x=self.hidden_layer0.output, in_dim=hidden_dim[0], out_dim=hidden_dim[1],
                                         activation=T.tanh, layer_id='h1')

        self.hidden_layer2 = HiddenLayer(x=self.hidden_layer1.output, in_dim=hidden_dim[1], out_dim=hidden_dim[2],
                                         activation=T.tanh, layer_id='h2')

        # final layer (softmax)
        self.softmax_layer = SoftmaxLayer(x=self.hidden_layer2.output, in_dim=hidden_dim[2], out_dim=n_classes,
                                          layer_id='s0')

        # Regularization
        # Regularization
        self.L1 = abs(self.hidden_layer0.weights).sum() + abs(self.hidden_layer1.weights).sum() + \
                  abs(self.hidden_layer2.weights).sum() + abs(self.softmax_layer.weights).sum()
        self.L2_sq = (self.hidden_layer0.weights ** 2).sum() + abs(self.hidden_layer1.weights ** 2).sum() + \
                     (self.hidden_layer2.weights ** 2).sum() + (self.softmax_layer.weights ** 2).sum()

        # output, errors, and likelihood
        self.y_predict = self.softmax_layer.y_predict
        self.negative_log_likelihood = self.softmax_layer.negative_log_likelihood
        self.errors = self.softmax_layer.errors
        self.output = self.softmax_layer.output
        self.params = self.hidden_layer0.params + self.hidden_layer1.params + self.hidden_layer2.params + \
                      self.softmax_layer.params

        self.type = "ReLUfourLayer"


class ConvolutionalNetwork3(Model):
    def __init__(self, x, in_dim, hidden_dim, n_classes,
                 batch_size, n_filters, n_channels, data_shape, filter_shape, poolsize):
        super(ConvolutionalNetwork3, self).__init__(x=x, in_dim=in_dim, n_classes=n_classes, hidden_dim=hidden_dim)

        # image shape is a 4D tensor, or a stack of data vectors, with dimensions
        # (batch size, # channels, width, height)
        input_shape = (batch_size, n_channels[0], data_shape[0], data_shape[1])
        layer0_input = x.reshape(input_shape)

        self.conv_layer = ConvPoolLayer(
            x=layer0_input,
            image_shape=input_shape,
            filter_shape=(n_filters[0], n_channels[0], filter_shape[0], filter_shape[1]),
            poolsize=poolsize,
            layer_id="c1"
        )

        conv_out_dims = (np.asarray([data_shape]) - np.asarray(filter_shape) + 1) / np.asarray(poolsize)

        assert(False not in (conv_out_dims >= 1)), "poolsize not appropriate for network"

        layer1_input = self.conv_layer.output.flatten(2)

        self.hidden_layer = HiddenLayer(
            x=layer1_input,
            in_dim=n_filters[0] * np.prod(conv_out_dims),
            out_dim=hidden_dim,
            layer_id="h2",
            activation=T.tanh,
        )

        self.softmax_layer = SoftmaxLayer(
            x=self.hidden_layer.output,
            in_dim=hidden_dim,
            out_dim=n_classes,
            layer_id="s1"
        )

        self.L1 = abs(self.conv_layer.weights).sum() + abs(self.hidden_layer.weights).sum() +\
                  abs(self.softmax_layer.weights).sum()

        self.L2_sq = abs(self.conv_layer.weights ** 2).sum() + abs(self.hidden_layer.weights ** 2).sum() +\
                     abs(self.softmax_layer.weights ** 2).sum()

        # output, errors, and likelihood
        self.y_predict = self.softmax_layer.y_predict
        self.negative_log_likelihood = self.softmax_layer.negative_log_likelihood
        self.errors = self.softmax_layer.errors
        self.output = self.softmax_layer.output
        self.params = self.conv_layer.params + self.hidden_layer.params + self.softmax_layer.params
        self.input = x
        self.type = "ConvNet3"


class VanillaNeuralNet(object):
    """[3], [2]
    A plain vanilla backprop neural network I made from scratch. See citations for the code that inspired
    this implementation
    """
    def __init__(self, input_dim, nb_classes, hidden_dims, activation_function):
        # eg. dimensions = [2, 10, 3] makes a 2-input, 10 hidden, 3 output NN
        # number of layers is the hidden node depth plus the input and output layers
        self.layers = len(hidden_dims) + 2
        dimensions = [input_dim] + hidden_dims + [nb_classes]
        np.random.seed(0)
        self.weights = [np.random.randn(x, y) / np.sqrt(x) for x, y, in izip(dimensions[:-1], dimensions[1:])]
        self.biases = [np.zeros((1, y)) for y in dimensions[1:]]
        self.activation = activation_function

    def predict_old(self, X):
        activation = X
        z = None
        # forward pass
        for bias, weight in izip(self.biases, self.weights):
            z = np.dot(activation, weight) + bias   # calculate input
            activation = self.activation(z, False)  # put though activation function
        # get softmax from final layer input
        exp_scores = np.exp(z)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        return probs

    def predict(self, x):
        activation = x
        z = None
        # forward pass
        i = 1
        for bias, weight in izip(self.biases, self.weights):
            z = np.dot(activation, weight) + bias   # calculate input
            # if we're in the hidden layer use the activation function
            if i < len(self.weights):
                activation = self.activation(z, False)  # put though activation function
                i += 1
                continue
            # if we're at the final 'output' layer, use the softmax
            else:
                assert (i == len(self.weights))
                exp_scores = np.exp(z)
                probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
                return probs

    def update_parameters(self, grad_weights, grad_biases, epsilon):
        # update based on learning rate (epsilon)
        self.weights = [w + -epsilon * dw for w, dw in izip(self.weights, grad_weights)]
        self.biases = [b + -epsilon * db for b, db in izip(self.biases, grad_biases)]

    def backprop(self, sample, label, lbda):
        # first do the forward pass, keeping track of everything
        zs = []                       # list to store z vectors
        X = sample.reshape(1, len(sample))
        activation = X          # initialize to input data
        activations = [X, ]     # list to store activations

        i = 1
        for bias, weight in izip(self.biases, self.weights):
            z = np.dot(activation, weight) + bias   # calculate input
            zs.append(z)                            # keep track
            if i < len(self.weights):
                activation = self.activation(z, False)  # put though activation function
                activations.append(activation)          # keep track
                i += 1
                continue
            else:
                activation = np.exp(z)
                activations.append(activation)

        # get softmax from final layer output
        probs = activations[-1] / np.sum(activations[-1], axis=1, keepdims=True)

        # backward pass
        delta = self.cost_derivate(probs, [label])

        # place to store gradients
        grad_w = [np.zeros(w.shape) for w in self.weights]
        grad_b = [np.zeros(b.shape) for b in self.biases]

        # initialize
        grad_w[-1] = np.dot(activations[-2].T, delta)
        grad_b[-1] = np.sum(delta, axis=0, keepdims=True)

        # backprop through the network, starting at the last hidden layer
        for layer in xrange(2, self.layers):
            delta = np.dot(delta, self.weights[-layer + 1].T) * self.activation(activations[-layer], True)
            grad_w[-layer] = np.dot(activations[-layer - 1].T, delta)
            grad_b[-layer] = np.sum(delta, axis=0)

        # regularize the gradient on the weights
        grad_w = [gw + lbda * w for gw, w in izip(grad_w, self.weights)]

        return grad_w, grad_b

    def fit(self, training_data, labels, epochs=10000, epsilon=0.01, lbda=0.01, print_loss=False):
        if print_loss is True:
            print("before training accuracy: %0.2f" % self.evaluate(training_data, labels), file=sys.stderr)
        for e in xrange(0, epochs):
            # first do the forward pass, keeping track of everything
            zs = []                       # list to store z vectors
            activation = training_data          # initialize to input data
            activations = [training_data, ]     # list to store activations

            i = 1
            for bias, weight in izip(self.biases, self.weights):
                z = np.dot(activation, weight) + bias   # calculate input
                zs.append(z)                            # keep track
                if i < len(self.weights):
                    activation = self.activation(z, False)  # put though activation function
                    activations.append(activation)          # keep track
                    i += 1
                    continue
                else:
                    activation = np.exp(z)
                    activations.append(activation)

            # get softmax from final layer output
            probs = activations[-1] / np.sum(activations[-1], axis=1, keepdims=True)

            # backward pass
            delta = self.cost_derivate(probs, labels)

            # place to store gradients
            grad_w = [np.zeros(w.shape) for w in self.weights]
            grad_b = [np.zeros(b.shape) for b in self.biases]

            # initialize
            grad_w[-1] = np.dot(activations[-2].T, delta)
            grad_b[-1] = np.sum(delta, axis=0, keepdims=True)

            # backprop through the network, starting at the last hidden layer
            for layer in xrange(2, self.layers):
                delta = np.dot(delta, self.weights[-layer + 1].T) * self.activation(activations[-layer], True)
                grad_w[-layer] = np.dot(activations[-layer - 1].T, delta)
                grad_b[-layer] = np.sum(delta, axis=0)

            # regularize the gradient on the weights
            grad_w = [gw + lbda * w for gw, w in izip(grad_w, self.weights)]

            self.update_parameters(grad_w, grad_b, epsilon)
            # update based on learning rate (epsilon)
            #self.weights = [w + -epsilon * dw for w, dw in izip(self.weights, grad_w)]
            #self.biases = [b + -epsilon * db for b, db in izip(self.biases, grad_b)]

            if print_loss and e % 1000 == 0:
                loss = self.calculate_loss(training_data, labels)
                accuracy = self.evaluate(training_data, labels)
                print("Loss after iteration %i: %f accuracy: %0.2f" % (e, loss, accuracy), file=sys.stderr)
        if print_loss is True:
            print("after training accuracy: %0.2f" % self.evaluate(training_data, labels), file=sys.stderr)

    def mini_batch_sgd(self, training_data, labels, epochs, batch_size, epsilon=0.01, lbda=0.01, print_loss=False):
        # place to store gradients
        whole_dataset = zip(training_data, labels)

        for e in xrange(0, epochs):
            n = len(whole_dataset)
            batches = [whole_dataset[k:k + batch_size] for k in xrange(0, n, batch_size)]

            for batch in batches:
                grad_w = [np.zeros(w.shape) for w in self.weights]
                grad_b = [np.zeros(b.shape) for b in self.biases]

                # get the gradient for each sample in the batch
                for sample, label in batch:
                    delta_w, delta_b = self.backprop(sample=sample, label=label, lbda=lbda)
                    #assert len()
                    grad_w = [dw + delta_w for dw, delta_w in izip(grad_w, delta_w)]
                    grad_b = [db + delta_b for db, delta_b in izip(grad_b, delta_b)]

                # update the parameters for this batch
                self.update_parameters(grad_w, grad_b, epsilon=epsilon)

            if e % 500 == 0 and print_loss == True:
                loss, accuracy = self.calculate_loss_and_accuracy(training_data, labels)
                print("Loss after iteration %i: %f accuracy: %0.2f" % (e, loss, accuracy), file=sys.stderr)

    def cost_derivate(self, output_probs, labels):
        output_probs[range(len(labels)), labels] -= 1
        return output_probs

    def calculate_loss(self, input_data, labels, reg_lambda=0.01):
        num_examples = len(input_data)
        assert len(input_data) == len(labels)

        probs = self.predict(input_data)

        # Calculating the loss
        corect_logprobs = -np.log(probs[range(num_examples), labels])
        data_loss = np.sum(corect_logprobs)

        # Add regulatization term to loss (optional)
        for w in self.weights:
            data_loss += 0.5 * reg_lambda * np.sum(np.square(w))

        return float(1. / num_examples * data_loss)

    def calculate_loss_and_accuracy(self, input_data, labels, reg_lambda=0.01):
        num_examples = len(input_data)
        assert len(input_data) == len(labels)

        probs = self.predict(input_data)

        hard_calls = np.argmax(probs, axis=1)
        accuracy = np.mean(hard_calls == labels)

        # Calculating the loss
        corect_logprobs = -np.log(probs[range(num_examples), labels])
        data_loss = np.sum(corect_logprobs)

        # Add regulatization term to loss (optional)
        for w in self.weights:
            data_loss += 0.5 * reg_lambda * np.sum(np.square(w))

        return float(1. / num_examples * data_loss), accuracy

    def evaluate(self, X, labels):
        probs = self.predict(X)
        hard_calls = np.argmax(probs, axis=1)
        return np.mean(hard_calls == labels)


