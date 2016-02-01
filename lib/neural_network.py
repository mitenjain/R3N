#!/usr/bin/env python
"""
Citations
[1]http://cs231n.github.io/neural-networks-case-study/
[2]http://nbviewer.ipython.org/github/dennybritz/nn-from-scratch/blob/master/nn-from-scratch.ipynb
[3]https://github.com/mnielsen/neural-networks-and-deep-learning
"""
from __future__ import print_function
import sys, os
import theano
import theano.tensor as T
import numpy as np
from utils import collect_data_vectors2, shuffle_and_maintain_labels, preprocess_data, chain, get_network, \
    stack_and_level_datasets2, stack_and_level_datasets3, append_and_level_labels2, append_and_level_labels3, \
    find_model_path
from optimization import mini_batch_sgd, mini_batch_sgd_with_annealing, cPickle


def predict(test_data, true_labels, batch_size, model, model_file=None):
    if model_file is not None:
        print("loading model from {}".format(model_file), end='\n', file=sys.stderr)
        model.load_from_file(file_path=model_file, careful=True)

    n_test_batches = test_data.shape[0] / batch_size

    y = T.ivector('y')

    prob_fcn = theano.function(inputs=[model.input],
                               outputs=model.output,
                               )

    error_fcn = theano.function(inputs=[model.input, y],
                                outputs=model.errors(y),
                                )
    errors = [error_fcn(test_data[x * batch_size: (x + 1) * batch_size],
                        true_labels[x * batch_size: (x + 1) * batch_size])
              for x in xrange(n_test_batches)]

    probs = [prob_fcn(test_data[x * batch_size: (x + 1) * batch_size])
             for x in xrange(n_test_batches)]

    probs = list(chain(*probs))

    return errors, probs


def evaluate_network(test_data, targets, model_file, model_type, batch_size, extra_args=None):
    # load the model file
    model = cPickle.load(open(model_file, 'r'))
    n_train_samples, data_dim = test_data.shape
    n_classes = len(set(targets))
    if data_dim != model['in_dim'] or n_classes != model['n_classes']:
        print("This data is not compatible with this network, exiting", file=sys.stderr)
        return False
    net = get_network(x=test_data, in_dim=model['in_dim'], n_classes=model['n_classes'], model_type=model_type,
                      hidden_dim=model['hidden_dim'], extra_args=extra_args)
    net.load_from_object(model=model, careful=True)
    errors, probs = predict(test_data=test_data, true_labels=targets, batch_size=batch_size, model=net, model_file=None)
    return errors, probs


def classify_with_network3(
        # alignment files
        group_1, group_2, group_3,  # these arguments should be strings that are used as the file suffix
        # which data to use
        strand, motif_start_positions, preprocess, events_per_pos, feature_set, title,
        # training params
        learning_algorithm, train_test_split, iterations, epochs, max_samples, batch_size,
        # model params
        learning_rate, L1_reg, L2_reg, hidden_dim, model_type, model_dir=None, extra_args=None,
        # output params
        out_path="./"):
    # checks and file IO
    assert(len(motif_start_positions) >= 3)
    out_file = open(out_path + title + ".tsv", 'wa')
    if model_dir is not None:
        print("looking for model in {}".format(os.path.abspath(model_dir)))
        model_file = find_model_path(os.path.abspath(model_dir), title)
    else:
        model_file = None
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

        nb_c_train, nb_c_xtr, nb_c_test = len(c_train), len(c_xtr), len(c_test)
        nb_mc_train, nb_mc_xtr, nb_mc_test = len(mc_train), len(mc_xtr), len(mc_test)
        nb_hmc_train, nb_hmc_xtr, nb_hmc_test = len(hmc_train), len(hmc_xtr), len(hmc_test)

        assert(nb_c_train > 0 and nb_mc_train > 0 and nb_hmc_train > 0), "got zero training vectors"

        # level training events so that the model gets equal exposure
        tr_level = np.min([nb_c_train, nb_mc_train, nb_hmc_train])
        xtr_level = np.min([nb_c_xtr, nb_mc_xtr, nb_hmc_xtr])
        test_level = np.min([nb_c_test, nb_mc_test, nb_hmc_test])

        # log how many vectors we got
        print("{motif}: got {C} C, {mC} mC, and {hmC} hmC, training vectors, leveled to {level}"
              .format(motif=title, C=nb_c_train, mC=nb_mc_train, hmC=nb_hmc_train, level=tr_level), file=sys.stderr)
        print("{motif}: got {xC} C, {xmC} mC, and {xhmC} hmC, cross-training vectors, leveled to {xlevel}"
              .format(motif=title, xC=nb_c_xtr, xmC=nb_mc_xtr, xhmC=nb_hmc_xtr, xlevel=xtr_level), file=sys.stderr)
        print("{motif}: got {xC} C, {xmC} mC, and {xhmC} hmC, test vectors, leveled to {tstLevel}"
              .format(motif=title, xC=len(c_test), xmC=len(mc_test), xhmC=len(hmc_test),
                      tstLevel=test_level), file=sys.stderr)

        # stack the data into one object
        # training data
        training_data = stack_and_level_datasets3(c_train, mc_train, hmc_train, tr_level)
        training_labels = append_and_level_labels3(c_tr_labels, mc_tr_labels, hmc_tr_labels, tr_level)

        # cross training
        xtrain_data = stack_and_level_datasets3(c_xtr, mc_xtr, hmc_xtr, xtr_level)
        xtrain_targets = append_and_level_labels3(c_xtr_targets, mc_xtr_targets, hmc_xtr_targets, xtr_level)

        # test
        test_data = stack_and_level_datasets3(c_test, mc_test, hmc_test, test_level)
        test_targets = append_and_level_labels3(c_test_targets, mc_test_targets, hmc_test_targets, test_level)

        prc_train, prc_xtrain, prc_test = preprocess_data(training_vectors=training_data,
                                                          xtrain_vectors=xtrain_data,
                                                          test_vectors=test_data,
                                                          preprocess=preprocess)

        #if evaluate is True:
        #    all_test_data = np.vstack((xtrain_data, test_data))
        #    all_test_targets = np.append(xtrain_targets, test_targets)
        #    errors, probs = evaluate_network(all_test_data, all_test_targets, model_dir, model_type, batch_size, extra_args)
        #    return

        # shuffle data
        X, y = shuffle_and_maintain_labels(prc_train, training_labels)

        working_directory_path = "{outpath}/{title}_Models/".format(outpath=out_path, title=title)
        if not os.path.exists(working_directory_path):
            os.makedirs(working_directory_path)
        trained_model_dir = "{workingdirpath}{iteration}/".format(workingdirpath=working_directory_path,
                                                                  iteration=i)

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

        errors, probs = predict(prc_test, test_targets, training_routine_args['batch_size'], net,
                                model_file=summary['best_model'])
        errors = 1 - np.mean(errors)
        probs = zip(probs, test_targets)

        print("{0}:{1}:{2} test accuracy.".format(title, i, (errors * 100)))
        out_file.write("{}\n".format(errors))
        scores.append(errors)

        with open("{}test_probs.pkl".format(trained_model_dir), 'w') as probs_file:
            cPickle.dump(probs, probs_file)

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
        learning_rate, L1_reg, L2_reg, hidden_dim, model_type, model_dir=None, extra_args=None,
        # output params
        out_path="./"):
    print("2 way classification")
    assert(len(motif_start_positions) >= 2)
    out_file = open(out_path + title + ".tsv", 'wa')
    if model_dir is not None:
        print("looking for model in {}".format(model_dir))
        model_file = find_model_path(model_dir, title)
    else:
        model_file = None

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
        for n, group in enumerate((group_1, group_2)):
            train_set, xtrain_set, test_set = collect_data_vectors2(label=n,
                                                                    files=group,
                                                                    motif_starts=motif_start_positions[n],
                                                                    dataset_title=title + "_group{}".format(n),
                                                                    **collect_data_vectors_args)
            add_to_list((train_set, xtrain_set, test_set))
        # unpack list
        g1_train, g1_tr_labels = list_of_datasets[0][0][0], list_of_datasets[0][0][1]
        g1_xtr, g1_xtr_targets = list_of_datasets[0][1][0], list_of_datasets[0][1][1]
        g1_test, g1_test_targets = list_of_datasets[0][2][0], list_of_datasets[0][2][1]

        g2_train, g2_tr_labels = list_of_datasets[1][0][0], list_of_datasets[1][0][1]
        g2_xtr, g2_xtr_targets = list_of_datasets[1][1][0], list_of_datasets[1][1][1]
        g2_test, g2_test_targets = list_of_datasets[1][2][0], list_of_datasets[1][2][1]

        nb_g1_train, nb_g1_xtr, nb_g1_test = len(g1_train), len(g1_xtr), len(g1_test)
        nb_g2_train, nb_g2_xtr, nb_g2_test = len(g2_train), len(g2_xtr), len(g2_test)
        assert(nb_g1_train > 0 and nb_g2_train > 0), "got {0} group 1 training and " \
                                                     "{1} group 2 training vectors".format(nb_g1_train, nb_g2_train)

        # level training and cross-training events so that the model gets equal exposure
        tr_level = np.min([nb_g1_train, nb_g2_train])
        xtr_level = np.min([nb_g1_xtr, nb_g2_xtr])
        test_level = np.min([nb_g1_test, nb_g2_test])
        print("{motif}: got {g1} group 1 and {g2} group 2 training vectors, leveled to {level}"
              .format(motif=title, g1=nb_g1_train, g2=nb_g2_train, level=tr_level))
        print("{motif}: got {g1} group 1 and {g2} group 2 cross-training vectors, leveled to {level}"
              .format(motif=title, g1=nb_g1_xtr, g2=nb_g2_xtr, level=xtr_level))
        print("{motif}: got {g1} group 1 and {g2} group 2 test vectors, leveled to {level}"
              .format(motif=title, g1=nb_g1_test, g2=nb_g2_test, level=test_level))

        training_data = stack_and_level_datasets2(g1_train, g2_train, tr_level)
        training_labels = append_and_level_labels2(g1_tr_labels, g2_tr_labels, tr_level)

        xtrain_data = stack_and_level_datasets2(g1_xtr, g2_xtr, xtr_level)
        xtrain_targets = append_and_level_labels2(g1_xtr_targets, g2_xtr_targets, xtr_level)

        test_data = stack_and_level_datasets2(g1_test, g2_test, test_level)
        test_targets = append_and_level_labels2(g1_test_targets, g2_test_targets, test_level)

        prc_train, prc_xtrain, prc_test = preprocess_data(training_vectors=training_data,
                                                          xtrain_vectors=xtrain_data,
                                                          test_vectors=test_data,
                                                          preprocess=preprocess)

        # evaluate

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

        errors, probs = predict(prc_test, test_targets, training_routine_args['batch_size'], net,
                                model_file=summary['best_model'])
        errors = 1 - np.mean(errors)
        print("{0}: {1} test accuracy.".format(title, (errors * 100)))
        out_file.write("{}\n".format(errors))
        scores.append(errors)

        with open("{}test_probs.pkl".format(trained_model_dir), 'w') as probs_file:
            cPickle.dump(probs, probs_file)

    print(">{motif}\t{accuracy}".format(motif=title, accuracy=np.mean(scores), end="\n"), file=out_file)
    return net


def test_error_distribution3(# alignment files
        group_1, group_2, group_3,  # these arguments should be strings that are used as the file suffix
        # which data to use
        strand, motif_start_positions, preprocess, events_per_pos, feature_set, title,
        # training params
        learning_algorithm, train_test_split, iterations, epochs, max_samples, batch_size,
        # model params
        learning_rate, L1_reg, L2_reg, hidden_dim, model_type, model_dir=None, extra_args=None,
        # output params
        out_path="./"):
    raise NotImplementedError
    # checks and file IO
    assert(len(motif_start_positions) >= 3)
    out_file = open(out_path + title + ".tsv", 'wa')

    scores = []

    # get the whole dataset for the 3 groups
    list_of_datasets = []
    add_to_list = list_of_datasets.append

    collect_data_vectors_args = {
        "events_per_pos": events_per_pos,
        "portion": train_test_split,
        "strand": strand,
        "max_samples": max_samples,
        "feature_set": feature_set,
        "kmer_length": 6,
        "split_dataset": False,
    }
    # makes a list of tuples [(dataset_1, labels_1),..., (dataset_3, labels_3)] for group 1-3
    for n, group in enumerate((group_1, group_2, group_3)):
        dataset, labels = collect_data_vectors2(label=n,
                                                files=group,
                                                motif_starts=motif_start_positions[n],
                                                dataset_title=title + "_group{}".format(n),
                                                **collect_data_vectors_args)
        add_to_list((dataset, labels))
    #
