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
from utils import collect_data_vectors2, shuffle_and_maintain_labels, preprocess_data
from optimization import mini_batch_sgd, mini_batch_sgd_with_annealing


def predict(test_data, true_labels, model, model_file=None):
    if model_file is not None:
        model.load(model_file)
    y = T.ivector('y')
    #test_data = test_data.reshape(3, 6)
    #predict_fcn = theano.function(inputs=[model.input],
    #                              outputs=model.y_predict,
    #                              )
    error_fcn = theano.function(inputs=[model.input, y],
                                outputs=model.errors(y),
                                )
    errors = error_fcn(test_data, true_labels)
    return errors


def predict2(test_data, true_labels, model, model_file=None):
    if model_file is not None:
        model.load(model_file)
    y = T.ivector('y')
    predict_fcn = theano.function(inputs=[model.input],
                                  outputs=model.y_predict,
                                  )
    prob_fcn = theano.function(inputs=[model.input],
                               outputs=model.output,
                               )
    error_fcn = theano.function(inputs=[model.input, y],
                                outputs=model.errors(y),
                                )
    errors = error_fcn(test_data, true_labels)

    predictions = predict_fcn(test_data)
    #print(predictions)
    probs = prob_fcn(test_data)
    for _ in zip(probs, predictions):
        print(">", _)
    #probs = probs[~np.isnan(probs).any(axis=1)]
    #for _ in probs:
    #    print(_)
    return errors


def classify_with_network3(
        # alignment files
        group_1, group_2, group_3,  # these arguments should be strings that are used as the file suffix
        # which data to use
        strand, motif_start_positions, preprocess, events_per_pos, feature_set, title,
        # training params
        learning_algorithm, train_test_split, iterations, epochs, max_samples, batch_size,
        # model params
        learning_rate, L1_reg, L2_reg, hidden_dim, model_type, model_file=None, extra_args=None,
        # output params
        out_path="./"):

    assert(len(motif_start_positions) >= 3)

    out_file = open(out_path + title + ".tsv", 'wa')

    # bin to hold accuracies for each iteration
    scores = []

    for i in xrange(iterations):
        c_train, c_tr_labels, c_xtr, c_xtr_targets = \
            collect_data_vectors2(events_per_pos=events_per_pos,
                                  label=0,
                                  portion=train_test_split,
                                  files=group_1,
                                  strand=strand,
                                  motif_starts=motif_start_positions[0],
                                  dataset_title=title+"_group1",
                                  max_samples=max_samples,
                                  feature_set=feature_set,
                                  kmer_length=6
                                  )

        mc_train, mc_tr_labels, mc_xtr, mc_xtr_targets = \
            collect_data_vectors2(events_per_pos=events_per_pos,
                                  label=1,
                                  portion=train_test_split,
                                  files=group_2,
                                  strand=strand,
                                  motif_starts=motif_start_positions[1],
                                  dataset_title=title+"_group2",
                                  max_samples=max_samples,
                                  feature_set=feature_set,
                                  kmer_length=6
                                  )

        hmc_train, hmc_tr_labels, hmc_xtr, hmc_xtr_targets = \
            collect_data_vectors2(events_per_pos=events_per_pos,
                                  label=2,
                                  portion=train_test_split,
                                  files=group_3,
                                  strand=strand,
                                  motif_starts=motif_start_positions[2],
                                  dataset_title=title+"_group3",
                                  max_samples=max_samples,
                                  feature_set=feature_set,
                                  kmer_length=6
                                  )

        nb_c_train, nb_c_xtr = len(c_train), len(c_xtr)
        nb_mc_train, nb_mc_xtr = len(mc_train), len(mc_xtr)
        nb_hmc_train, nb_hmc_xtr = len(hmc_train), len(hmc_xtr)

        assert(nb_c_train > 0 and nb_mc_train > 0 and nb_hmc_train > 0), "got zero training vectors"

        # level training events so that the model gets equal exposure
        tr_level = np.min([nb_c_train, nb_mc_train, nb_hmc_train])
        xtr_level = np.min([nb_c_xtr, nb_mc_xtr, nb_hmc_xtr])

        print("{motif}: got {C} C, {mC} mC, and {hmC} hmC, training vectors, leveled to {level}"
              .format(motif=title, C=nb_c_train, mC=nb_mc_train,
                      hmC=nb_hmc_train, level=tr_level))
        print("{motif}: got {xC} C, {xmC} mC, and {xhmC} hmC, cross-training vectors, leveled to {xlevel}"
              .format(motif=title, xC=nb_c_xtr, xmC=nb_mc_xtr, xhmC=nb_hmc_xtr, xlevel=xtr_level))

        # level C training and cross training data
        c_train = c_train[:tr_level]
        c_tr_labels = c_tr_labels[0][:tr_level]
        c_xtr = c_xtr[:xtr_level]
        c_xtr_targets = c_xtr_targets[0][:xtr_level]

        # level mC training and cross training data
        mc_train = mc_train[:tr_level]
        mc_tr_labels = mc_tr_labels[0][:tr_level]
        mc_xtr = mc_xtr[:xtr_level]
        mc_xtr_targets = mc_xtr_targets[0][:xtr_level]

        # level hmC training and cross training data
        hmc_train = hmc_train[:tr_level]
        hmc_tr_labels = hmc_tr_labels[0][:tr_level]
        hmc_xtr = hmc_xtr[:xtr_level]
        hmc_xtr_targets = hmc_xtr_targets[0][:xtr_level]

        # stack the data into one object TODO could do the leveling here
        training_data = np.vstack((c_train, mc_train, hmc_train))
        training_targets = np.append(c_tr_labels, np.append(mc_tr_labels, hmc_tr_labels))
        xtrain_data = np.vstack((c_xtr, mc_xtr, hmc_xtr))
        xtrain_targets = np.append(c_xtr_targets, np.append(mc_xtr_targets, hmc_xtr_targets))

        prc_train, prc_xtrain = preprocess_data(training_vectors=training_data,
                                                test_vectors=xtrain_data,
                                                preprocess=preprocess)

        X, y = shuffle_and_maintain_labels(prc_train, training_targets)

        trained_model_dir = "{0}{1}_Models/".format(out_path, title)

        if learning_algorithm == "annealing":
            net, summary = mini_batch_sgd_with_annealing(motif=title, train_data=X, labels=y,
                                                xTrain_data=xtrain_data, xTrain_labels=xtrain_targets,
                                                learning_rate=learning_rate, L1_reg=L1_reg, L2_reg=L2_reg,
                                                epochs=epochs, batch_size=batch_size, hidden_dim=hidden_dim,
                                                model_type=model_type, model_file=model_file, extra_args=extra_args,
                                                trained_model_dir=trained_model_dir)
        else:
            net, summary = mini_batch_sgd(motif=title, train_data=X, labels=y,
                                          xTrain_data=xtrain_data, xTrain_labels=xtrain_targets,
                                          learning_rate=learning_rate, L1_reg=L1_reg, L2_reg=L2_reg,
                                          epochs=epochs, batch_size=batch_size, hidden_dim=hidden_dim,
                                          model_type=model_type, model_file=model_file, extra_args=extra_args,
                                          trained_model_dir=trained_model_dir)

        errors = predict(xtrain_data, xtrain_targets, net)
        errors = 1 - errors
        out_file.write("{}\n".format(errors))
        scores.append(errors)

    print(">{motif}\t{accuracy}".format(motif=title, accuracy=np.mean(scores), end="\n"), file=out_file)
    return net


def classify_with_network2(
        # alignment files
        group_1, group_2, group_3,
        # which data to use
        strand, motif_start_positions, preprocess, events_per_pos, feature_set, title,
        # training params
        learning_algorithm, train_test_split, iterations, epochs, max_samples, batch_size,
        # model params
        learning_rate, L1_reg, L2_reg, hidden_dim, model_type, model_file=None, extra_args=None,
        # output params
        out_path="./"):

    assert(len(motif_start_positions) >= 2)

    out_file = open(out_path + title + ".tsv", 'wa')

    # bin to hold accuracies for each iteration
    scores = []

    for i in xrange(iterations):
        g1_train, g1_tr_labels, g1_xtr, g1_xtr_targets = \
            collect_data_vectors2(events_per_pos=events_per_pos,
                                  label=0,
                                  portion=train_test_split,
                                  files=group_1,
                                  strand=strand,
                                  motif_starts=motif_start_positions[0],
                                  dataset_title=title+"_group1",
                                  max_samples=max_samples,
                                  feature_set=feature_set,
                                  kmer_length=6
                                  )

        g2_train, g2_tr_labels, g2_xtr, g2_xtr_targets = \
            collect_data_vectors2(events_per_pos=events_per_pos,
                                  label=1,
                                  portion=train_test_split,
                                  files=group_2,
                                  strand=strand,
                                  motif_starts=motif_start_positions[1],
                                  dataset_title=title+"_group2",
                                  max_samples=max_samples,
                                  feature_set=feature_set,
                                  kmer_length=6
                                  )

        nb_g1_train, nb_g1_xtr = len(g1_train), len(g1_xtr)
        nb_g2_train, nb_g2_xtr = len(g2_train), len(g2_xtr)
        assert(nb_g1_train > 0 and nb_g2_train > 0), "got {0} group 1 training and " \
                                                     "{1} group 2 training vectors".format(nb_g1_train, nb_g2_train)

        # level training and cross-training events so that the model gets equal exposure
        tr_level = np.min([nb_g1_train, nb_g2_train])
        xtr_level = np.min([nb_g1_xtr, nb_g2_xtr])
        print("{motif}: got {g1} group 1 and {g2} group 2 training vectors, leveled to {level}"
              .format(motif=title, g1=nb_g1_train, g2=nb_g2_train, level=tr_level))
        print("{motif}: got {g1} group 1 and {g2} group 2 cross-training vectors, leveled to {level}"
              .format(motif=title, g1=nb_g1_xtr, g2=nb_g2_xtr, level=xtr_level))

        g1_train = g1_train[:tr_level]
        g1_tr_labels = g1_tr_labels[0][:tr_level]

        g2_train = g2_train[:tr_level]
        g2_tr_labels = g2_tr_labels[0][:tr_level]

        g1_xtr = g1_xtr[:xtr_level]
        g1_xtr_targets = g1_xtr_targets[:xtr_level]

        g2_xtr = g2_xtr[:xtr_level]
        g2_xtr_targets = g2_xtr_targets[:xtr_level]

        # stack the data into one object
        training_data = np.vstack((g1_train, g2_train))
        training_targets = np.append(g1_tr_labels, g2_tr_labels)
        xtrain_data = np.vstack((g1_xtr, g2_xtr))
        xtrain_targets = np.append(g1_xtr_targets, g2_xtr_targets)

        prc_train, prc_xtrain = preprocess_data(training_vectors=training_data,
                                                test_vectors=xtrain_data,
                                                preprocess=preprocess)

        X, y = shuffle_and_maintain_labels(prc_train, training_targets)

        trained_model_dir = "{0}{1}_Models/".format(out_path, title)

        if learning_algorithm == "annealing":
            net, summary = mini_batch_sgd_with_annealing(motif=title, train_data=X, labels=y,
                                                xTrain_data=xtrain_data, xTrain_labels=xtrain_targets,
                                                learning_rate=learning_rate, L1_reg=L1_reg, L2_reg=L2_reg,
                                                epochs=epochs, batch_size=batch_size, hidden_dim=hidden_dim,
                                                model_type=model_type, model_file=model_file, extra_args=extra_args,
                                                trained_model_dir=trained_model_dir)
        else:
            net, summary = mini_batch_sgd(motif=title, train_data=X, labels=y,
                                 xTrain_data=xtrain_data, xTrain_labels=xtrain_targets,
                                 learning_rate=learning_rate, L1_reg=L1_reg, L2_reg=L2_reg,
                                 epochs=epochs, batch_size=batch_size, hidden_dim=hidden_dim,
                                 model_type=model_type, model_file=model_file, extra_args=extra_args,
                                 trained_model_dir=trained_model_dir)

        errors = predict(xtrain_data, xtrain_targets, net)
        errors = 1 - errors
        out_file.write("{}\n".format(errors))
        scores.append(errors)

    print(">{motif}\t{accuracy}".format(motif=title, accuracy=np.mean(scores), end="\n"), file=out_file)
    return net
