#!/usr/bin/env python
"""
Citations
[1]http://cs231n.github.io/neural-networks-case-study/
[2]http://nbviewer.ipython.org/github/dennybritz/nn-from-scratch/blob/master/nn-from-scratch.ipynb
[3]https://github.com/mnielsen/neural-networks-and-deep-learning
"""
from __future__ import print_function
import sys
import theano
import theano.tensor as T
import numpy as np
from utils import collect_data_vectors, shuffle_and_maintain_labels
from optimization import mini_batch_sgd, mini_batch_sgd_with_annealing


def predict(test_data, true_labels, model, model_file=None):
    if model_file is not None:
        model.load(model_file)

    y = T.ivector('y')

    #predict_fcn = theano.function(inputs=[model.input],
    #                              outputs=model.y_predict,
    #                              )

    error_fcn = theano.function(inputs=[model.input, y],
                                outputs=model.errors(y),
                                )

    #predictions = predict_fcn(test_data)
    errors = error_fcn(test_data, true_labels)

    #print("prediction", predictions)
    #print("errors", errors)
    return errors


def classify_with_network2(
        # alignments
        c_alignments, mc_alignments, hmc_alignments,
        # which alignments to go/get
        forward, motif_start_position, no_center,
        # training params
        train_test_split, iterations, epochs, max_samples, batch_size,
        # model params
        learning_rate, L1_reg, L2_reg, hidden_dim, model_type, model_file=None,
        # output params
        print_loss=False, out_path="./"):

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

        # routine to center data features

        # get the mean
        feature_mean = np.nanmean(training_data, axis=0)

        # center
        centered_training_data = training_data - feature_mean

        # convert NaNs to zeros
        centered_training_data = np.nan_to_num(centered_training_data)

        X, y = shuffle_and_maintain_labels(centered_training_data, labels)

        c_targets = np.zeros(len(c_test), dtype=np.int32)

        mc_targets = np.zeros(len(mc_test), dtype=np.int32)
        mc_targets.fill(1)

        hmc_targets = np.zeros(len(hmc_test), dtype=np.int32)
        hmc_targets.fill(2)

        all_test_data = np.vstack((c_test, mc_test, hmc_test))
        all_test_data -= feature_mean
        all_test_data = np.nan_to_num(all_test_data)

        all_targets = np.concatenate((c_targets, mc_targets, hmc_targets))

        trained_model_dir = "{0}{1}_Models/".format(out_path, motif_start_position)

        net = mini_batch_sgd_with_annealing(train_data=X, labels=y,
                             xTrain_data=all_test_data, xTrain_labels=all_targets,
                             learning_rate=learning_rate, L1_reg=L1_reg, L2_reg=L2_reg,
                             epochs=epochs, batch_size=batch_size, hidden_dim=hidden_dim,
                             model_type=model_type, model_file=model_file,
                             trained_model_dir=trained_model_dir)

        errors = predict(all_test_data, all_targets, net)
        errors = 1 - errors
        out_file.write("{}\n".format(errors))
        scores.append(errors)

    print(">{motif}\t{accuracy}".format(motif=motif_start_position, accuracy=np.mean(scores), end="\n"), file=out_file)
    return net


