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
from logistic_sgd import load_data
import theano
import theano.tensor as T

# globals
rng = np.random.RandomState()


class NeuralNetwork(object):
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


class SoftmaxLayer(object):
    """Multiclass logistic regression layer class
    """
    def __init__(self, x, in_dim, out_dim):
        self.weights = theano.shared(value=np.zeros([in_dim, out_dim], dtype=theano.config.floatX),
                                     name='weights',
                                     borrow=True
                                     )
        self.biases = theano.shared(value=np.zeros([out_dim], dtype=theano.config.floatX),
                                    name='biases',
                                    borrow=True
                                    )
        self.params = [self.weights, self.biases]

        self.input = x
        self.output = self.prob_y_given_x(x)
        self.y_predict = T.argmax(self.output, axis=1)

    def prob_y_given_x(self, input_data):
        return T.nnet.softmax(T.dot(input_data, self.weights) + self.biases)

    def negative_log_likelihood(self, labels):
        return -T.mean(T.log(self.output)[T.arange(labels.shape[0]), labels])

    def errors(self, labels):
        return T.mean(T.neq(self.y_predict, labels))


class HiddenLayer(object):
    def __init__(self, x, in_dim, out_dim, W=None, b=None, activation=T.tanh):
        if W is None:
            W_values = np.asarray(rng.uniform(low=-np.sqrt(6. / (in_dim + out_dim)),
                                              high=np.sqrt(6. / (in_dim + out_dim)),
                                              size=(in_dim, out_dim)),
                                  dtype=theano.config.floatX
                                  )
            W = theano.shared(value=W_values, name='W', borrow=True)
        if b is None:
            b_values = np.zeros((out_dim,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.weights = W
        self.biases = b
        self.params = [self.weights, self.biases]

        #self.input = x
        lin_out = T.dot(x, self.weights) + self.biases
        self.output = lin_out if activation is None else activation(lin_out)


class fastNeuralNetwork(object):
    def __init__(self, x, in_dim, hidden_dim, n_classes):
        # first layer (hidden)
        self.hidden_layer = HiddenLayer(x=x, in_dim=in_dim, out_dim=hidden_dim, activation=T.tanh)
        # final layer (softmax)
        self.softmax_layer = SoftmaxLayer(x=self.hidden_layer.output, in_dim=hidden_dim, out_dim=n_classes)
        self.L1 = abs(self.hidden_layer.weights).sum() + abs(self.softmax_layer.weights).sum()
        self.L2_sq = (self.hidden_layer.weights ** 2).sum() + (self.softmax_layer.weights ** 2).sum()
        #self.L2_sq = sum((self.hidden_layer.weights ** 2)) + sum((self.softmax_layer.weights ** 2))
        self.negative_log_likelihood = self.softmax_layer.negative_log_likelihood
        self.errors = self.softmax_layer.errors
        self.params = self.hidden_layer.params + self.softmax_layer.params
        self.input = x


def minibatch_sgd(train_data, labels, valid_data, valid_labels,
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

    net = fastNeuralNetwork(x=x, in_dim=64, n_classes=10, hidden_dim=500)

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
               for param, nambla_param in zip (net.params, nambla_params)]

    # main function? could make this an attribute and reduce redundant code
    train_fcn = theano.function(inputs=[batch_index],
                                outputs=cost,
                                updates=updates,
                                givens={
                                    x: train_set_x[batch_index * batch_size: (batch_index + 1) * batch_size],
                                    y: train_set_y[batch_index * batch_size: (batch_index + 1) * batch_size]
                                })

    # train the model
    for epoch in xrange(0, epochs):
        for i in xrange(n_train_batches):
            batch_avg_cost = train_fcn(i)
        if epoch % 500 == 0:
            valid_costs = [valid_fcn(_) for _ in xrange(n_valid_batches)]
            mean_validation_cost = 100 * (1 - np.mean(valid_costs))
            print("At epoch {0}, accuracy {1}".format(epoch, mean_validation_cost))

    return net


def shared_dataset(data_x, data_y, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """

        shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')
