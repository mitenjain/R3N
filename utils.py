#!/usr/bin/env python
"""Utility functions for MinION signal alignments
"""
from __future__ import print_function
import os
import theano
import sys
import numpy as np
import theano.tensor as T
from model import NeuralNetwork, FastNeuralNetwork, ThreeLayerNetwork
from random import shuffle


def get_motif_range(ref_start, forward, reference_length=891):
    kmer_length = 6
    if forward:
        template_motif_range = range(ref_start, ref_start + kmer_length)
        return template_motif_range
    if not forward:
        complement_motif_range = range(ref_start, ref_start + kmer_length)
        return complement_motif_range


def cull_motif_features(start, tsv, forward):
    # load the tsv
    data = np.loadtxt(tsv, dtype=str)
    motif_range = get_motif_range(start, forward)

    # build a feature vector that has the first 6 elements as the template features and the second
    # six elements as the complement features, the features are selected as the ones with the maximum
    # posterior probability
    feature_vector = np.empty(18)
    feature_vector.fill(np.nan)

    # to keep track of maximum
    feature_posteriors = np.zeros([1, 6])

    for line in data:
        if line[4] == "t" and int(line[0]) in motif_range and forward is True:
            # determine which event in the motif this is
            # array has format: [e0, p0, e1, p1, e2, p2, e3, p3, e4, p4, e5, p5]
            # multiply by 2 to index through the array
            e_index = motif_range.index(int(line[0]))
            vector_index = e_index * 3
            delta_mean = float(line[5]) - float(line[9])
            delta_noise = float(line[6]) - float(line[10])  # change in noise not used yet
            posterior = float(line[8])

            # if the posterior for this event is higher than the one we have previously seen,
            if posterior > feature_posteriors[0, e_index]:
                feature_vector[vector_index] = delta_mean
                feature_vector[vector_index + 1] = delta_noise
                feature_vector[vector_index + 2] = posterior
                feature_posteriors[0, e_index] = posterior
        if line[4] == "c" and int(line[0]) in motif_range and forward is False:
            e_index = motif_range.index(int(line[0]))
            vector_index = e_index * 3
            delta_mean = float(line[5]) - float(line[9])
            delta_noise = float(line[6]) - float(line[10])
            posterior = line[8]

            if posterior > feature_posteriors[0, e_index]:
                feature_vector[vector_index] = delta_mean
                feature_vector[vector_index + 1] = delta_noise
                feature_vector[vector_index + 2] = posterior
                feature_posteriors[0, e_index] = posterior

    return feature_vector


def collect_data_vectors(path, forward, labels, label, portion, motif_start, max_samples):
    # collect the files
    if forward:
        tsvs = [x for x in os.listdir(path) if x.endswith(".forward.tsv") and os.stat(path + x).st_size != 0]
    else:
        tsvs = [x for x in os.listdir(path) if x.endswith(".backward.tsv") and os.stat(path + x).st_size != 0]

    # shuffle
    shuffle(tsvs)

    if max_samples < len(tsvs):
        tsvs = tsvs[:max_samples]

    # get the number of files we're going to use
    split_index = int(portion * len(tsvs))

    # container for training and test data
    # data vector is 6 events and 6 posteriors
    train_data = np.zeros([split_index, 18])
    test_data = np.zeros([len(tsvs) - split_index, 18])

    for i, f in enumerate(tsvs[:split_index]):
        vector = cull_motif_features(motif_start, path + f, forward)
        train_data[i:i + 1] = vector
        labels.append(label)  # TODO move this out of the function

    for i, f in enumerate(tsvs[split_index:]):
        vector = cull_motif_features(motif_start, path + f, forward)
        test_data[i:i+1] = vector

    return train_data, labels, test_data


def shuffle_and_maintain_labels(data, labels):
    assert len(data) == len(labels)
    dataset = zip(data, labels)
    shuffle(dataset)
    X = [x[0] for x in dataset]
    y = [x[1] for x in dataset]

    return np.asarray(X), y


def center_features_on_training_data(training_vectors, test_vectors):
    feature_mean_vector = np.nanmean(training_vectors, axis=0)

    training_vectors -= feature_mean_vector
    test_vectors -= feature_mean_vector
    return training_vectors, test_vectors


def get_network(x, in_dim, n_classes, hidden_dim, type):
    if type == "twoLayer":
        return FastNeuralNetwork(x=x, in_dim=in_dim, n_classes=n_classes, hidden_dim=hidden_dim)
    if type == "threeLayer":
        return ThreeLayerNetwork(x=x, in_dim=in_dim, n_classes=n_classes, hidden_dim=hidden_dim)
    else:
        print("Invalid model type", file=sys.stderr)
        return False


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




