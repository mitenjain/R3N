#!/usr/bin/env python

from toy_datasets import *
from neural_network import *
from sklearn import preprocessing
from utils import *

from itertools import chain
from optimization import *

import pandas as pd
import timeit

# generating test data
#X, Y = generate_2_class_moon_data()
#X2, Y2 = generate_2_class_moon_data()
#X, Y = generate_3_class_spiral_data(nb_classes=3, theta=0.5, plot=False)
#X2, Y2 = generate_3_class_spiral_data(nb_classes=3, theta=0.5, plot=False)
#X, y = load_iris_dataset()

# digit dataset
'''
train, test = load_digit_dataset(1000, 0.1)
X = np.array([x[0] for x in train])
Y = [y[1] for y in train]
Y = np.asarray(Y)
X2 = np.array([x[0] for x in test])
Y2 = [y[1] for y in test]
'''

tsv_t = "../marginAlign/cPecan/tests/temp/tempFiles_alignment/" \
        "makeson_PC_MA_286_R7.3_ZYMO_C_1_09_11_15_1714_1_ch1_file1_strand.vl.forward.tsv"

tsv_o = "../marginAlign/cPecan/tests/temp/signalalign-v-1230/" \
        "makeson_PC_MA_286_R7.3_ZYMO_C_1_09_11_15_1714_1_ch1_file1_strand.vl.forward.tsv"


aln = "../marginAlign/cPecan/tests/temp/tempFiles_alignment/*.tsv"
aln2 = "../marginAlign/cPecan/tests/test_alignments/newf_conditional_model/C/tempFiles_alignment/*.forward.tsv"

m = [300, 747]
dst = "all"
strand = "t"

#features = cull_motif_features4(m, tsv_t, strand, feature_set=dst, kmer_length=6)
#print features

#events = []
#for strand in ["t", "c"]:
#    events += list(chain(
#                         *features.ix[(features['ref_pos'] == 300) & (features['strand'] == strand)]
#                         .drop(['ref_pos', 'strand'], 1)[:1].values.tolist()))


tr, xtr, ts = collect_data_vectors2(events_per_pos=1,
                                    label=0,
                                    portion=0.5,
                                    files=aln,
                                    strand=strand,
                                    motif_starts=m,
                                    dataset_title="test",
                                    max_samples=10,
                                    feature_set=dst)

print "training", tr[0], "\n"
print "cross-training", xtr[0], "\n"
print "testing", ts[0], "\n"

