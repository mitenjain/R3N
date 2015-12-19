#!/usr/bin/env python
"""Utility functions for MinION signal alignments
"""
from __future__ import print_function
import os
from neural_network import *
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
    #feature_vector = np.zeros(12)
    feature_vector = np.empty(12)
    feature_vector.fill(np.nan)

    feature_posteriors = np.zeros([1, 6])  # to keep track of maximum

    for line in data:
        if line[4] == "t" and int(line[0]) in motif_range and forward is True:
            # determine which event in the motif this is
            # array has format: [e0, p0, e1, p1, e2, p2, e3, p3, e4, p4, e5, p5]
            # multiply by 2 to index through the array
            e_index = motif_range.index(int(line[0]))
            vector_index = e_index * 2
            delta_mean = float(line[5]) - float(line[9])
            # delta_noise = float(line[6]) - float(line[10])  # change in noise not used yet
            posterior = float(line[8])

            # if the posterior for this event is higher than the one we have previously seen,
            if posterior > feature_posteriors[0, e_index]:
                feature_vector[vector_index] = delta_mean
                feature_vector[vector_index + 1] = posterior
                feature_posteriors[0, e_index] = posterior
        if line[4] == "c" and int(line[0]) in motif_range and forward is False:
            e_index = motif_range.index(int(line[0]))
            vector_index = e_index * 2
            delta_mean = float(line[5]) - float(line[9])
            delta_noise = float(line[6]) - float(line[10])
            posterior = line[8]

            if posterior > feature_posteriors[e_index]:
                feature_vector[vector_index] = delta_mean
                feature_vector[vector_index + 1] = posterior
                feature_posteriors[e_index] = posterior

    #if zero_center is True:
    #    feature_mean = np.mean([feature_vector[i] for i in xrange(0, len(feature_vector), 2)
    #                            if feature_vector[i] != 0])
    #    for i in xrange(0, len(feature_vector), 2):
    #        if feature_vector[i] != 0.0:
    #            feature_vector[i] -= feature_mean
    #return np.nan_to_num(feature_vector)
    return feature_vector


def collect_data_vectors(path, forward, labels, label, portion, motif_start, max_samples):
    """collects the training data
    """
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
    train_data = np.zeros([split_index, 12])
    test_data = np.zeros([len(tsvs) - split_index, 12])

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
    #centered_training = training_vectors - feature_mean_vector
    #centered_test = test_vectors - feature_mean_vector
    training_vectors -= feature_mean_vector
    test_vectors -= feature_mean_vector
    return training_vectors, test_vectors


def classify_with_network(c_alignments, mc_alignments, hmc_alignments,
                          forward, motif_start_position, center_data,
                          train_test_split, iterations, epochs, max_samples, mini_batch,
                          activation_function, epsilon, lbda, hidden_shape, print_loss,
                          out_path):
    if forward:
        direction_label = ".forward"
    else:
        direction_label = ".backward"

    out_file = open(out_path + str(motif_start_position) + direction_label + ".tsv", 'wa')

    # bin to hold accuracies for each iteration
    scores = []

    for i in xrange(iterations):
        labels = []
        c_train, labels, c_test = collect_data_vectors(c_alignments, forward, labels, 0,
                                                       train_test_split, motif_start_position,
                                                       max_samples)
        mc_train, labels, mc_test = collect_data_vectors(mc_alignments, forward, labels, 1,
                                                         train_test_split, motif_start_position,
                                                         max_samples)
        hmc_train, labels, hmc_test = collect_data_vectors(hmc_alignments, forward, labels, 2,
                                                           train_test_split, motif_start_position,
                                                           max_samples)
        training_data = np.vstack((c_train, mc_train, hmc_train))

        ## added feature centering  ## TODO make optional
        feature_mean = np.nanmean(training_data, axis=0)

        centered_training_data = training_data - feature_mean

        centered_training_data = np.nan_to_num(centered_training_data)

        X, y = shuffle_and_maintain_labels(centered_training_data, labels)

        net = NeuralNetwork(input_dim=X.shape[1],
                            nb_classes=len(set(y)),
                            hidden_dims=[10],
                            activation_function=activation_function)

        if mini_batch is not None:
            net.mini_batch_sgd(training_data=X,
                               labels=y,
                               epochs=epochs,
                               batch_size=mini_batch,
                               epsilon=epsilon,
                               lbda=lbda,
                               print_loss=print_loss)

        else:
            net.fit(training_data=X,
                    labels=y,
                    epochs=epochs,
                    epsilon=epsilon,
                    lbda=lbda,
                    print_loss=print_loss)

        c_targets = np.zeros(len(c_test))

        mc_targets = np.zeros(len(mc_test))
        mc_targets.fill(1)

        hmc_targets = np.zeros(len(hmc_test))
        hmc_targets.fill(2)

        all_test_data = np.vstack((c_test, mc_test, hmc_test))
        all_test_data -= feature_mean
        all_test_data = np.nan_to_num(all_test_data)

        all_targets = np.concatenate((c_targets, mc_targets, hmc_targets))

        accuracy = net.evaluate(all_test_data, all_targets)
        print(accuracy, file=out_file)
        scores.append(accuracy)

    print(">{motif}\t{accuracy}".format(motif=motif_start_position, accuracy=np.mean(scores), end="\n"), file=out_file)
    return






