#!/usr/bin/env python
"""Utility functions for MinION signal alignments
"""
from __future__ import print_function
import os
import theano
import sys
import numpy as np
import theano.tensor as T
from itertools import chain
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
    """Used to cull all of the aligned features in Echelon alignments
    """
    # load the tsv
    data = np.loadtxt(tsv, dtype=str)
    motif_range = get_motif_range(start, forward)

    # build a feature vector that has the first 6 elements as the template features and the second
    # six elements as the complement features, the features are selected as the ones with the maximum
    # posterior probability
    feature_dict = {}

    for line in data:
        if line[4] == "t" and int(line[0]) in motif_range and forward is True:
            delta_mean = float(line[5]) - float(line[9])
            posterior = float(line[8])
            try:
                feature_dict[line[0]].append((delta_mean, posterior))
            except KeyError:
                feature_dict[line[0]] = [(delta_mean, posterior)]

        if line[4] == "c" and int(line[0]) in motif_range and forward is False:
            delta_mean = float(line[5]) - float(line[9])
            posterior = line[8]
            try:
                feature_dict[line[0]].append((delta_mean, posterior))
            except KeyError:
                feature_dict[line[0]] = [(delta_mean, posterior)]

    return feature_dict


def get_vectors(events_per_position, path, tsvs, motif_start, forward, split_idx):
    nb_event_features = 2  # mean diff and noise
    nb_positions = 6  # 6-mers
    vector_size = events_per_position * nb_event_features * nb_positions
    position_idx_offset = events_per_position * nb_event_features
    # containers
    train_data = []
    tr_append = train_data.append
    xtrain_data = []
    xt_append = xtrain_data.append
    motif_range = get_motif_range(motif_start, forward)

    # todo make this into it's own function
    for i, f in enumerate(tsvs[:split_idx]):
        # get the dictionary of events aligned to each position
        motif_dict = cull_motif_features(motif_start, path + f, forward)
        if motif_dict.keys() == []:
            continue
        vect = np.full(shape=vector_size, fill_value=np.nan)
        for idx, position in enumerate(motif_range):
            # sort the events in by decending posterior match prob, only take the first so many, and then
            # turn the list of tuples into a list of floats
            try:
                events = list(chain(
                        *sorted(motif_dict[str(position)], key=lambda e: e[1], reverse=True)[:nb_events_per_column]))
                # add them to the feature vector
                for _ in xrange(len(events)):
                    #train_data[i, ((idx * position_idx_offset) + _)] = events[_]
                    vect[(idx * position_idx_offset) + _] = events[_]
            except KeyError:
                continue
        tr_append(vect)



def collect_deep_data_vectors(events_per_pos, path, forward, label, portion, motif_start, max_samples):
    # collect the files
    if forward:
        tsvs = [x for x in os.listdir(path) if x.endswith(".forward.tsv") and os.stat(path + x).st_size != 0]
    else:
        tsvs = [x for x in os.listdir(path) if x.endswith(".backward.tsv") and os.stat(path + x).st_size != 0]

    # shuffle
    shuffle(tsvs)

    assert(portion <= 1.0 and max_samples >= 1)

    if max_samples < len(tsvs):
        tsvs = tsvs[:max_samples]

    # get the number of files we're going to use
    split_index = int(portion * len(tsvs))

    # container for training and test data
    # for the echelon alignments, we allow for a defined number  aligned events, each event has
    # two features (diff. mean, and posterior), there are 6 positions, so for each read (set of
    # observations) we need:
    # nb_events * nb_event_features * positions
    nb_events_per_column = events_per_pos
    nb_event_features = 2
    nb_positions = 6
    # precomputed
    vector_size = nb_events_per_column * nb_event_features * nb_positions
    position_idx_offset = nb_events_per_column * nb_event_features
    # containers
    train_data = []
    tr_append = train_data.append
    xtrain_data = []
    xt_append = xtrain_data.append
    motif_range = get_motif_range(motif_start, forward)

    # todo make this into it's own function
    for i, f in enumerate(tsvs[:split_index]):
        # get the dictionary of events aligned to each position
        motif_dict = cull_motif_features(motif_start, path + f, forward)
        if motif_dict.keys() == []:
            continue
        vect = np.full(shape=vector_size, fill_value=np.nan)
        for idx, position in enumerate(motif_range):
            # sort the events in by decending posterior match prob, only take the first so many, and then
            # turn the list of tuples into a list of floats
            try:
                events = list(chain(
                        *sorted(motif_dict[str(position)], key=lambda e: e[1], reverse=True)[:nb_events_per_column]))
                # add them to the feature vector
                for _ in xrange(len(events)):
                    #train_data[i, ((idx * position_idx_offset) + _)] = events[_]
                    vect[(idx * position_idx_offset) + _] = events[_]
            except KeyError:
                continue
        tr_append(vect)

    for i, f in enumerate(tsvs[split_index:]):
        # get the dictionary of events aligned to each position
        motif_dict = cull_motif_features(motif_start, path + f, forward)
        if motif_dict.keys() == []:
            continue
        vect = np.full(shape=vector_size, fill_value=np.nan)
        for idx, position in enumerate(motif_range):
            # sort the events in by decending posterior match prob, only take the first so many, and then
            # turn the list of tuples into a list of floats
            try:
                events = list(chain(
                        *sorted(motif_dict[str(position)], key=lambda e: e[1], reverse=True)[:nb_events_per_column]))
                # add them to the feature vector
                for _ in xrange(len(events)):
                    #train_data[i, ((idx * position_idx_offset) + _)] = events[_]
                    vect[(idx * position_idx_offset) + _] = events[_]
            except KeyError:
                continue
        xt_append(vect)

    train_labels = np.full(shape=[1, len(train_data)], fill_value=label, dtype=np.int32)
    xtrain_labels = np.full(shape=[1, len(xtrain_data)], fill_value=label, dtype=np.int32)

    return np.asarray(train_data), train_labels, np.asarray(xtrain_data), xtrain_labels


def shuffle_and_maintain_labels(data, labels):
    assert len(data) == len(labels)
    dataset = zip(data, labels)
    shuffle(dataset)
    X = [x[0] for x in dataset]
    y = [x[1] for x in dataset]

    return np.asarray(X), y


def preprocess_data_only_events(training_vectors, test_vectors, preprocess=None):
    assert(len(training_vectors.shape) == 2)
    if preprocess == "center" or preprocess == "normalize":
        training_mean_vector = np.nanmean(training_vectors, axis=0)
        # don't center the posteriors
        for i in xrange(0, 12, 2):
            training_mean_vector[(i + 1)] = 0
        training_vectors -= training_mean_vector
        test_vectors -= training_mean_vector

        if preprocess == "normalize":
            training_std_vector = np.nanstd(training_vectors, axis=0)
            # don't norm posteriors
            for i in xrange(0, 12, 2):
                training_std_vector[(i + 1)] = 1
            training_vectors /= training_std_vector
            test_vectors /= training_std_vector

    prc_training_vectors = np.nan_to_num(training_vectors)
    prc_test_vectors = np.nan_to_num(test_vectors)

    return prc_training_vectors, prc_test_vectors


def preprocess_data(training_vectors, test_vectors, preprocess=None):
    assert(len(training_vectors.shape) == 2 and len(test_vectors.shape) == 2)
    if preprocess == "center" or preprocess == "normalize":
        training_mean_vector = np.nanmean(training_vectors, axis=0)
        training_vectors -= training_mean_vector
        test_vectors -= training_mean_vector

        if preprocess == "normalize":
            training_std_vector = np.nanstd(training_vectors, axis=0)
            training_vectors /= training_std_vector
            test_vectors /= training_std_vector

    prc_training_vectors = np.nan_to_num(training_vectors)
    prc_test_vectors = np.nan_to_num(test_vectors)

    return prc_training_vectors, prc_test_vectors


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




