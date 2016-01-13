#!/usr/bin/env python
"""Run a SVM on collected alignment data
"""
import sys
from neural_network import classify_with_network3
from argparse import ArgumentParser
from multiprocessing import Process, current_process, Manager


def parse_args():
    parser = ArgumentParser(description=__doc__)

    # query files
    parser.add_argument('--C_files', '-c', action='store',
                        dest='c_files', required=True, type=str, default=None,
                        help="directory with C files")
    parser.add_argument('--mC_files', '-mc', action='store',
                        dest='mc_files', required=True, type=str, default=None,
                        help="directory with mC files")
    parser.add_argument('--hmC_files', '-hmc', action='store',
                        dest='hmc_files', required=True, type=str, default=None,
                        help="directory with hmC files")
    parser.add_argument('--strand', '-st', action='store', dest='strand', required=True,
                        type=str, help="which strand to use, options = {t, c, both}")
    parser.add_argument('-nb_files', '-nb', action='store', dest='nb_files', required=False,
                        default=50, type=int, help="maximum number of reads to use")
    parser.add_argument('--jobs', '-j', action='store', dest='jobs', required=False,
                        default=4, type=int, help="number of jobs to run concurrently")
    parser.add_argument('--iter', '-i', action='store', dest='iter', required=False,
                        default=1, type=int, help="number of iterations to do")
    parser.add_argument('--learning_algorithm', '-a', dest='learning_algo', required=False,
                        default=None, action='store', type=str, help="options: \"annealing\"")
    parser.add_argument('--epochs', '-ep', action='store', dest='epochs', required=False,
                        default=10000, type=int, help="number of iterations to do")
    parser.add_argument('--batch_size', '-b', action='store', dest='batch_size', required=True, type=int,
                        default=None, help='specify batch size')
    parser.add_argument('--learning_rate', '-e', action='store', dest='learning_rate',
                        required=False, default=0.01, type=float)
    parser.add_argument('--L1_reg', '-L1', action='store', dest='L1', required=False,
                        default=0.0, type=float)
    parser.add_argument('--L2_reg', '-L2', action='store', dest='L2', required=False,
                        default=0.001, type=float)
    parser.add_argument('--train_test', '-s', action='store', dest='split', required=False,
                        default=0.9, type=float, help="train/test split")
    parser.add_argument('--preprocess', '-p', action='store', required=False, default=None,
                        dest='preprocess', help="options:\nnormalize\ncenter\ndefault:None")
    parser.add_argument("--feature_set", '-f', action='store', dest='features', required=False,
                        type=str, default=None, help="pick features: all, mean, noise, default: mean with"
                                                     " posteriors")
    parser.add_argument('--events', '-ev', action='store', required=True, dest='events', type=int,
                        help='number of events per alignment column to use')
    parser.add_argument('--null', action='store_true', dest='null', required=False, default=False,
                        help="classify null motifs")
    parser.add_argument('--title', action='store', dest='title', type=str,
                        required=False, default="default title")
    parser.add_argument('--output_location', '-o', action='store', dest='out',
                        required=True, type=str, default=None,
                        help="directory to put results")
    args = parser.parse_args()
    return args


def run_nn(work_queue, done_queue):
    #networks = []
    try:
        for f in iter(work_queue.get, 'STOP'):
            #classify_with_network(**f)
            n = classify_with_network3(**f)
            #networks.append(n)
    except Exception:
        done_queue.put("%s failed" % current_process().name)


def main(args):
    args = parse_args()

    # Change network here
    net_shape = [100, 100]
    net_type = "ReLUthreeLayer"

    assert (args.features in [None, "mean", "noise", "all"]), "invalid feature subset selection"

    start_message = """
#    Starting Neural Net analysis for {title}.
#    Command line: {cmd}
#    Looking at {nbFiles} files.
#    Using events from strand {strand}
#    Network type: {type}
#    Network dims: {dims}
#    Learning algorithm: {algo}
#    Collecting {nb_events} events per reference position.
#    Batch size: {batch}
#    Non-default feature set: {feature_set}
#    Iterations: {iter}.
#    Epochs: {epochs}
#    Data pre-processing: {center}
#    Train/test split: {train_test}
#    L1 reg: {L1}
#    L2 reg: {L2}
#    Output to: {out}""".format(nbFiles=args.nb_files, strand=args.strand, iter=args.iter,
                                train_test=args.split, out=args.out, epochs=args.epochs, center=args.preprocess,
                                L1=args.L1, L2=args.L2, type=net_type, dims=net_shape, nb_events=args.events,
                                cmd=" ".join(sys.argv[:]), batch=args.batch_size, algo=args.learning_algo,
                                feature_set=args.features, title=args.title)

    print >> sys.stdout, start_message

    if args.null is True:
        #motifs = [11, 62, 87, 218, 295, 371, 383, 457, 518, 740, 785, 805, 842, 866]
        #motifs = [11, 62, 87]
        motifs = [[11, 62, 87, 218, 295, 371, 383, 457, 518, 740, 785, 805, 842, 866]]
    else:
        #motifs = [747, 354, 148, 796, 289, 363, 755, 626, 813, 653, 525, 80, 874]
        #motifs = [[747], [354]]
        motifs = [[747, 354, 148, 796, 289, 363, 755, 626, 813, 653, 525, 80, 874]]

    workers = args.jobs
    work_queue = Manager().Queue()
    done_queue = Manager().Queue()
    jobs = []

    for motif in motifs:
        nn_args = {
            "group_1": args.c_files,
            "group_2": args.mc_files,
            "group_3": args.hmc_files,
            "strand": args.strand,
            "motif_start_positions": motif,
            "preprocess": args.preprocess,
            "events_per_pos": args.events,
            "feature_set": args.features,
            "title": args.title + " ",
            "learning_algorithm": args.learning_algo,
            "train_test_split": args.split,
            "iterations": args.iter,
            "epochs": args.epochs,
            "max_samples": args.nb_files,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "L1_reg": args.L1,
            "L2_reg": args.L2,
            "hidden_dim": net_shape,  # temp hardcoded
            "model_type": net_type,  # temp hardcoded
            "out_path": args.out,
        }
        #classify_with_network3(**nn_args)  # activate for debugging
        work_queue.put(nn_args)

    for w in xrange(workers):
        p = Process(target=run_nn, args=(work_queue, done_queue))
        p.start()
        jobs.append(p)
        work_queue.put('STOP')

    for p in jobs:
        p.join()

    done_queue.put('STOP')

    print >> sys.stderr, "\n\tFinished Neural Net"
    print >> sys.stdout, "\n\tFinished Neural Net"


if __name__ == "__main__":
    sys.exit(main(sys.argv))
