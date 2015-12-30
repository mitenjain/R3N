#!/usr/bin/env python
"""Utility functions for MinION signal alignments
"""
from __future__ import print_function
import os
import theano
import sys
import pandas as pd
import numpy as np
import theano.tensor as T
from itertools import chain
from model import NeuralNetwork, FastNeuralNetwork, ThreeLayerNetwork, ReLUThreeLayerNetwork
from random import shuffle


def get_motif_range(ref_start, forward, reference_length=891):
    kmer_length = 6
    if forward:
        template_motif_range = range(ref_start, ref_start + kmer_length)
        return template_motif_range
    if not forward:
        complement_motif_range = range(ref_start, ref_start + kmer_length)
        return complement_motif_range


def cull_motif_features2(motif, tsv, forward=True, kmer_length=6):
    if forward:
        strand = "t"
    else:
        strand = "c"
    try:
        data = pd.read_table(tsv, usecols=(0, 1, 4, 5, 6, 8, 9, 10),
                             dtype={'ref_pos': np.int32, 'event_idx': np.int32, 'strand': np.str,
                                    'event_mean': np.float64, 'event_noise': np.float64,
                                    'prob': np.float64, 'E_mean': np.float64,
                                    'E_noise': np.float64},
                             header=None,
                             names=['ref_pos', 'event_idx', 'strand', 'event_mean',
                                    'event_noise', 'prob', 'E_mean', 'E_noise'])
        motif_range = range(motif, motif + kmer_length)

        motif_rows = data.ix[(data['ref_pos'].isin(motif_range)) & (data['strand'] == strand)]

        features = pd.DataFrame({"ref_pos": motif_rows['ref_pos'],
                                 "delta_mean": motif_rows['event_mean'] - motif_rows['E_mean'],
                                 "posterior": motif_rows['prob']})

        if features.empty:
            return False

        f = features.sort_values(['ref_pos', 'posterior'], ascending=[True, False])\
            .drop_duplicates(subset='delta_mean')
        return f

    except:
        return False


def cull_motif_features(start, tsv, forward):
    """Used to cull all of the aligned features in Echelon alignments
    """
    # load the tsv, checking for broken files
    try:
        data = np.loadtxt(tsv, dtype=str)
    except ValueError:
        return False

    motif_range = get_motif_range(start, forward)

    # build a feature vector that has the first 6 elements as the template features and the second
    # six elements as the complement features, the features are selected as the ones with the maximum
    # posterior probability
    feature_dict = {}

    for line in data:
        if line[4] == "t" and int(line[0]) in motif_range and forward is True:
            delta_mean = float(line[5]) - float(line[9])
            #delta_mean = float(line[5]) / float(line[9])  # turn on to try quotient
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
    # if we didn't find any events, return false
    if feature_dict.keys == []:
        return False

    return feature_dict


def collect_data_vectors2(events_per_pos, path, forward, label, portion, motif_start, max_samples, kmer_length=6):
    # collect the files
    if forward:
        tsvs = [x for x in os.listdir(path) if x.endswith(".forward.tsv") and os.stat(path + x).st_size != 0]
    else:
        tsvs = [x for x in os.listdir(path) if x.endswith(".backward.tsv") and os.stat(path + x).st_size != 0]

    # shuffle
    shuffle(tsvs)

    assert(portion < 1.0 and max_samples >= 1)

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

    print("{0}: Getting vectors from {1}".format(motif_start, path), end='\n', file=sys.stderr)

    for i, f in enumerate(tsvs):
        # get the dictionary of events aligned to each position
        motif_table = cull_motif_features2(motif_start, path + f, forward)
        if motif_table is False:
            continue
        vect = np.full(shape=vector_size, fill_value=np.nan)
        for idx, position in enumerate(xrange(motif_start, motif_start + kmer_length)):
            # sort the events in by decending posterior match prob, only take the first so many, and then
            # turn the list of tuples into a list of floats
            try:
                # this giant thing takes the DataFrame which has all of the events aligned to the portion we're looking
                # at, gets only the ones aligned to 'position', removes the ref_position column, takes only the highest
                # events_per_pos events, turns it into a list of lists, then chains it into one long list. in case you
                # forgot, the table comes pre-sorted, so when you take the top n events, they are already sorted
                # in ascending posterior match prob
                events = list(chain(
                    *motif_table.ix[motif_table['ref_pos'] == position]
                    .drop('ref_pos', 1)[:events_per_pos].values.tolist()))
                # add them to the feature vector
                for _ in xrange(len(events)):
                    vect[(idx * position_idx_offset) + _] = events[_]
            except KeyError:
                continue
        if i < split_index:
            tr_append(vect)
        else:
            xt_append(vect)

    train_labels = np.full(shape=[1, len(train_data)], fill_value=label, dtype=np.int32)
    xtrain_labels = np.full(shape=[1, len(xtrain_data)], fill_value=label, dtype=np.int32)

    #print("{0}: got {1} training and {2} cross-training vectors for label {3}".format(
    #        motif_start, len(train_data), len(xtrain_data), label),
    #      file=sys.stderr)

    return np.asarray(train_data), train_labels, np.asarray(xtrain_data), xtrain_labels


def collect_data_vectors(events_per_pos, path, forward, label, portion, motif_start, max_samples):
    # collect the files
    if forward:
        tsvs = [x for x in os.listdir(path) if x.endswith(".forward.tsv") and os.stat(path + x).st_size != 0]
    else:
        tsvs = [x for x in os.listdir(path) if x.endswith(".backward.tsv") and os.stat(path + x).st_size != 0]

    # shuffle
    shuffle(tsvs)

    #assert(portion < 1.0 and max_samples >= 1)

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

    print("{0}: Getting vectors from {1}".format(motif_start, path), end='\n', file=sys.stderr)

    for i, f in enumerate(tsvs):
        # get the dictionary of events aligned to each position
        motif_dict = cull_motif_features(motif_start, path + f, forward)
        if motif_dict is False:
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
                    vect[(idx * position_idx_offset) + _] = events[_]
            except KeyError:
                continue
        if i < split_index:
            tr_append(vect)
        else:
            xt_append(vect)

    train_labels = np.full(shape=[1, len(train_data)], fill_value=label, dtype=np.int32)
    xtrain_labels = np.full(shape=[1, len(xtrain_data)], fill_value=label, dtype=np.int32)

    print("{0}: got {1} training and {2} cross-training vectors for label {3}".format(
            motif_start, len(train_data), len(xtrain_data), label),
          file=sys.stderr)

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
    if type == "ReLUthreeLayer":
        return ReLUThreeLayerNetwork(x=x, in_dim=in_dim, n_classes=n_classes, hidden_dim=hidden_dim)
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




