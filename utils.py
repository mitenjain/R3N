#!/usr/bin/env python
"""Utility functions for MinION signal alignments
"""
import os
from neural_network import *


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
    feature_vector = np.zeros(12)
    feature_posteriors = np.zeros(6)  # to keep track of maximum

    for line in data:
        if line[4] == "t" and int(line[0]) in motif_range and forward is True:
            # determine which event in the motif this is
            # array has format: [e0, p0, e1, p1, e2, p2, e3, p3, e4, p4, e5, p5]
            # multiply by 2 to index through the array
            e_index = motif_range.index(int(line[0]))
            vector_index = e_index * 2
            delta_mean = float(line[5]) - float(line[9])
            # delta_noise = float(line[6]) - float(line[10])  # change in noise not used yet
            posterior = line[8]

            # if the posterior for this event is higher than the one we have previously seen,
            if posterior > feature_posteriors[e_index]:
                feature_vector[vector_index] = delta_mean
                feature_vector[vector_index + 1] = posterior
                feature_posteriors[e_index] = posterior
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
        weight, vector = cull_motif_features(motif_start, path + f, forward)
        test_data[i:i+1] = vector

    return train_data, labels, test_data