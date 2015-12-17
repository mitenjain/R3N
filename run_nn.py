#!/usr/bin/env python
"""Run a SVM on collected alignment data
"""
import sys
from utils import *
from activation_functions import *
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
    parser.add_argument('--backward', '-bw', action='store_false', dest='forward',
                        default=True, help='forward mapped reads?')
    parser.add_argument('-nb_files', '-nb', action='store', dest='nb_files', required=False,
                        default=50, type=int, help="maximum number of reads to use")
    parser.add_argument('--jobs', '-j', action='store', dest='jobs', required=False,
                        default=4, type=int, help="number of jobs to run concurrently")
    parser.add_argument('--iter', '-i', action='store', dest='iter', required=False,
                        default=2, type=int, help="number of iterations to do")
    parser.add_argument('--epochs', '-ep', action='store', dest='epochs', required=False,
                        default=10000, type=int, help="number of iterations to do")
    parser.add_argument('--mini_batch', '-b', action='store', dest='mini_batch', required=False, type=int,
                        default=None, help='specify size of mini-batches for mini-batch training')
    parser.add_argument('--epsilon', '-e', action='store', dest='epsilon',
                        required=False, default=0.01, type=float)
    parser.add_argument('--lambda', '-l', action='store', dest='lbda', required=False,
                        default=0.01, type=float)
    parser.add_argument('--train_test', '-s', action='store', dest='split', required=False,
                        default=0.9, type=float, help="train/test split")
    parser.add_argument('--print_loss', '-lo', action='store_true', dest='print_loss',
                        default=False, help='print loss during training?')
    parser.add_argument('--output_location', '-o', action='store', dest='out',
                        required=True, type=str, default=None,
                        help="directory to put results")
    args = parser.parse_args()
    return args


def run_nn(work_queue, done_queue):
    try:
        for f in iter(work_queue.get, 'STOP'):
            classify_with_network(**f)
    except Exception:
        done_queue.put("%s failed" % current_process().name)


def main(args):
    args = parse_args()

    start_message = """
#    Command line: {cmd}
#    Starting Neural Net analysis.
#    Looking at {nbFiles} files.
#    Forward mapped strand: {forward}.
#    Iterations: {iter}.
#    Train/test split: {train_test}
#    Output to: {out}""".format(nbFiles=args.nb_files, forward=args.forward, iter=args.iter,
                                train_test=args.split, out=args.out,
                                cmd=" ".join(sys.argv[:]))

    print >> sys.stdout, start_message

    motifs = [747, 354, 148, 796, 289, 363, 755, 626, 813, 653, 525, 80, 874]
    #motifs = [747, 354]

    workers = args.jobs
    work_queue = Manager().Queue()
    done_queue = Manager().Queue()
    jobs = []

    for motif in motifs:
        nn_args = {
            "c_alignments": args.c_files,
            "mc_alignments": args.mc_files,
            "hmc_alignments": args.hmc_files,
            "forward": args.forward,
            "motif_start_position": motif,
            "train_test_split": args.split,
            "iterations": args.iter,
            "epochs": args.epochs,
            "max_samples": args.nb_files,
            "mini_batch": args.mini_batch,
            "activation_function": hyperbolic_tangent,
            "epsilon": args.epsilon,
            "lbda": args.lbda,
            "hidden_shape": [10],
            "print_loss": args.print_loss,
            "out_path": args.out,
        }
        #classify_with_network(**nn_args)  # activate for debugging
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


if __name__ == "__main__":
    sys.exit(main(sys.argv))
