#!/usr/bin/env python
import sys
import glob
import os
import cPickle
sys.path.append("../")
from lib.utils import cull_motif_features4
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser(description=__doc__)

    # query files
    parser.add_argument('--files', '-d', action='store',
                        dest='files', required=True, type=str, default=None,
                        help="directory with alignment files")
    parser.add_argument('--strand', '-st', action='store', dest='strand', required=True,
                        help="which strand get get stats for")
    parser.add_argument('--output_location', '-o', action='store', dest='out',
                        required=True, type=str, default=None,
                        help="file to put results")
    args = parser.parse_args()
    return args


def main(args):
    args = parse_args()

    # motif sites
    motifs = [80, 148, 289, 354, 363, 525, 626, 653, 747, 755, 796, 813, 874]
    # null sites
    nulls = [11, 62, 87, 218, 295, 371, 383, 457, 518, 740, 785, 805, 842, 866]
    all_sites = motifs + nulls

    # get the files
    tsvs = [x for x in glob.glob(args.files) if os.stat(x).st_size != 0]
    # container for stats
    stats = {}
    strand = args.strand
    for tsv in tsvs:
        motif_table = cull_motif_features4(motif=all_sites, tsv=tsv, strand=strand, feature_set="mean")
        if motif_table is False:
            continue
        for row in motif_table.iterrows():
            try:
                idx = str(row[1]['ref_pos'])
                d_mean = row[1]['delta_mean']
                stats[idx].append(d_mean)
            except KeyError:
                idx = str(row[1]['ref_pos'])
                d_mean = row[1]['delta_mean']
                stats[idx] = [d_mean]
    cPickle.dump(stats, open(args.out, 'w'))


if __name__ == "__main__":
    sys.exit(main(sys.argv))


