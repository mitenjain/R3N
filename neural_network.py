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
from utils import collect_data_vectors2, shuffle_and_maintain_labels, preprocess_data, shared_dataset
from optimization import mini_batch_sgd, mini_batch_sgd_with_annealing


def predict(test_data, true_labels, batch_size, model, model_file=None):
    if model_file is not None:
        model.load(model_file)

    n_test_batches = test_data.shape[0] / batch_size

    y = T.ivector('y')

    #predict_fcn = theano.function(inputs=[model.input],
    #                              outputs=model.y_predict,
    #                              )
    error_fcn = theano.function(inputs=[model.input, y],
                                outputs=model.errors(y),
                                )

    errors = [error_fcn(test_data[x * batch_size: (x + 1) * batch_size],
                        true_labels[x * batch_size: (x + 1) * batch_size]
                        ) for x in xrange(n_test_batches)]

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

    collect_data_vectors_args = {
        "events_per_pos": events_per_pos,
        "portion": train_test_split,
        "strand": strand,
        "max_samples": max_samples,
        "feature_set": feature_set,
        "kmer_length": 6
    }

    for i in xrange(iterations):
        list_of_datasets = []  # [((g1, g1l), (xg1, xg1l), (tg1, tg1l)), ... ]
        add_to_list = list_of_datasets.append
        for n, group in enumerate((group_1, group_2, group_3)):
            train_set, xtrain_set, test_set = collect_data_vectors2(label=n,
                                                                    files=group,
                                                                    motif_starts=motif_start_positions[n],
                                                                    dataset_title=title + "_group{}".format(n),
                                                                    **collect_data_vectors_args)
            add_to_list((train_set, xtrain_set, test_set))

        # unpack to make things easier, list_of_datasets[group][set_idx][vector/labels]
        c_train, c_tr_labels = list_of_datasets[0][0][0], list_of_datasets[0][0][1]
        c_xtr, c_xtr_targets = list_of_datasets[0][1][0], list_of_datasets[0][1][1]
        c_test, c_test_targets = list_of_datasets[0][2][0], list_of_datasets[0][2][1]

        mc_train, mc_tr_labels = list_of_datasets[1][0][0], list_of_datasets[1][0][1]
        mc_xtr, mc_xtr_targets = list_of_datasets[1][1][0], list_of_datasets[1][1][1]
        mc_test, mc_test_targets = list_of_datasets[1][2][0], list_of_datasets[1][2][1]

        hmc_train, hmc_tr_labels = list_of_datasets[2][0][0], list_of_datasets[2][0][1]
        hmc_xtr, hmc_xtr_targets = list_of_datasets[2][1][0], list_of_datasets[2][1][1]
        hmc_test, hmc_test_targets = list_of_datasets[2][2][0], list_of_datasets[2][2][1]

        nb_c_train, nb_c_xtr = len(c_train), len(c_xtr)
        nb_mc_train, nb_mc_xtr = len(mc_train), len(mc_xtr)
        nb_hmc_train, nb_hmc_xtr = len(hmc_train), len(hmc_xtr)

        assert(nb_c_train > 0 and nb_mc_train > 0 and nb_hmc_train > 0), "got zero training vectors"

        # level training events so that the model gets equal exposure
        tr_level = np.min([nb_c_train, nb_mc_train, nb_hmc_train])
        xtr_level = np.min([nb_c_xtr, nb_mc_xtr, nb_hmc_xtr])

        # log how many vectors we got
        print("{motif}: got {C} C, {mC} mC, and {hmC} hmC, training vectors, leveled to {level}"
              .format(motif=title, C=nb_c_train, mC=nb_mc_train, hmC=nb_hmc_train, level=tr_level), file=sys.stderr)
        print("{motif}: got {xC} C, {xmC} mC, and {xhmC} hmC, cross-training vectors, leveled to {xlevel}"
              .format(motif=title, xC=nb_c_xtr, xmC=nb_mc_xtr, xhmC=nb_hmc_xtr, xlevel=xtr_level), file=sys.stderr)
        print("{motif}: got {xC} C, {xmC} mC, and {xhmC} hmC, test vectors"
              .format(motif=title, xC=len(c_test), xmC=len(mc_test), xhmC=len(hmc_test)), file=sys.stderr)

        # stack the data into one object
        # training data
        training_data = np.vstack((c_train[:tr_level], mc_train[:tr_level], hmc_train[:tr_level]))
        training_labels = np.append(c_tr_labels[:tr_level], np.append(mc_tr_labels[:tr_level],
                                                                      hmc_tr_labels[:tr_level]))
        # cross training
        xtrain_data = np.vstack((c_xtr[:xtr_level], mc_xtr[:xtr_level], hmc_xtr[:xtr_level]))
        xtrain_targets = np.append(c_xtr_targets[:xtr_level], np.append(mc_xtr_targets[:xtr_level],
                                                                        hmc_xtr_targets[:xtr_level]))
        # test
        test_data = np.vstack((c_test, mc_test, hmc_test))
        test_targets = np.append(c_test_targets, np.append(mc_test_targets, hmc_test_targets))

        prc_train, prc_xtrain, prc_test = preprocess_data(training_vectors=training_data,
                                                          xtrain_vectors=xtrain_data,
                                                          test_vectors=test_data,
                                                          preprocess=preprocess)

        # shuffle data
        X, y = shuffle_and_maintain_labels(prc_train, training_labels)

        trained_model_dir = "{0}{1}_Models/".format(out_path, title)

        training_routine_args = {
            "motif": title,
            "train_data": X,
            "labels": y,
            "xTrain_data": prc_xtrain,
            "xTrain_targets": xtrain_targets,
            "learning_rate": learning_rate,
            "L1_reg": L1_reg,
            "L2_reg": L2_reg,
            "epochs": epochs,
            "batch_size": batch_size,
            "hidden_dim": hidden_dim,
            "model_type": model_type,
            "model_file": model_file,
            "trained_model_dir": trained_model_dir,
            "extra_args": extra_args
        }

        if learning_algorithm == "annealing":
            net, summary = mini_batch_sgd_with_annealing(**training_routine_args)
        else:
            net, summary = mini_batch_sgd(**training_routine_args)

        errors = predict(prc_test, test_targets, training_routine_args['batch_size'], net)
        errors = np.mean(errors)
        errors = 1 - errors
        print("{0}: {1} test error.".format(title, errors))
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
    raise NotImplementedError

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
                                                xtrain_vectors=xtrain_data,
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
