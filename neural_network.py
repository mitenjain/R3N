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
from utils import collect_data_vectors, shuffle_and_maintain_labels, preprocess_data
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
    errors = error_fcn(test_data, true_labels)
    return errors


def classify_with_network2(
        # alignment files
        c_alignments, mc_alignments, hmc_alignments,
        # which data to use
        forward, motif_start_position, preprocess, events_per_pos,
        # training params
        learning_algorithm, train_test_split, iterations, epochs, max_samples, batch_size,
        # model params
        learning_rate, L1_reg, L2_reg, hidden_dim, model_type, model_file=None,
        # output params
        out_path="./"):

    if forward:
        direction_label = ".forward"
    else:
        direction_label = ".backward"

    out_file = open(out_path + str(motif_start_position) + direction_label + ".tsv", 'wa')

    # bin to hold accuracies for each iteration
    scores = []

    for i in xrange(iterations):
        c_train, c_tr_labels, c_test, c_xtr_targets = \
            collect_data_vectors(events_per_pos, c_alignments, forward, 0,
                                      train_test_split, motif_start_position, max_samples)

        mc_train, mc_tr_labels, mc_test, mc_xtr_targets = \
            collect_data_vectors(events_per_pos, mc_alignments, forward, 1,
                                      train_test_split, motif_start_position, max_samples)

        hmc_train, hmc_tr_labels, hmc_test, hmc_xtr_targets = \
            collect_data_vectors(events_per_pos, hmc_alignments, forward, 2,
                                      train_test_split, motif_start_position, max_samples)

        assert(len(c_test) > 0 and len(mc_test) > 0 and len(hmc_test) > 0)
        # stack the data into one object
        training_data = np.vstack((c_train, mc_train, hmc_train))
        training_targets = np.append(c_tr_labels, np.append(mc_tr_labels, hmc_tr_labels))
        xtrain_data = np.vstack((c_test, mc_test, hmc_test))
        xtrain_targets = np.append(c_xtr_targets, np.append(mc_xtr_targets, hmc_xtr_targets))

        prc_train, prc_xtrain = preprocess_data(training_vectors=training_data,
                                                test_vectors=xtrain_data,
                                                preprocess=preprocess)

        X, y = shuffle_and_maintain_labels(prc_train, training_targets)

        trained_model_dir = "{0}{1}_Models/".format(out_path, motif_start_position)

        if learning_algorithm == "annealing":
            net = mini_batch_sgd_with_annealing(train_data=X, labels=y,
                                                xTrain_data=xtrain_data, xTrain_labels=xtrain_targets,
                                                learning_rate=learning_rate, L1_reg=L1_reg, L2_reg=L2_reg,
                                                epochs=epochs, batch_size=batch_size, hidden_dim=hidden_dim,
                                                model_type=model_type, model_file=model_file,
                                                trained_model_dir=trained_model_dir)
        else:
            net = mini_batch_sgd(train_data=X, labels=y,
                                 xTrain_data=xtrain_data, xTrain_labels=xtrain_targets,
                                 learning_rate=learning_rate, L1_reg=L1_reg, L2_reg=L2_reg,
                                 epochs=epochs, batch_size=batch_size, hidden_dim=hidden_dim,
                                 model_type=model_type, model_file=model_file,
                                 trained_model_dir=trained_model_dir)

        errors = predict(xtrain_data, xtrain_targets, net)
        errors = 1 - errors
        out_file.write("{}\n".format(errors))
        scores.append(errors)

    print(">{motif}\t{accuracy}".format(motif=motif_start_position, accuracy=np.mean(scores), end="\n"), file=out_file)
    return net


