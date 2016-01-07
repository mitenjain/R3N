#!/usr/bin/env python

from toy_datasets import *
from neural_network import *
from sklearn import preprocessing
from utils import *
from activation_functions import *
from itertools import chain
from optimization import *
import input_data
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
# Testing Neural Nets #

tsv_t = "../marginAlign/cPecan/tests/temp/tempFiles_alignment/" \
        "makeson_PC_MA_286_R7.3_ZYMO_C_1_09_11_15_1714_1_ch1_file1_strand.vl.forward.tsv"

tsv_o = "../marginAlign/cPecan/tests/temp/signalalign-v-1230/" \
        "makeson_PC_MA_286_R7.3_ZYMO_C_1_09_11_15_1714_1_ch1_file1_strand.vl.forward.tsv"


aln = "../marginAlign/cPecan/tests/temp/tempFiles_alignment/"

m = 300
dst = "all"

features = cull_motif_features4(m, tsv_t, feature_set=dst, forward=True, kmer_length=6)
print features

features_2 = cull_motif_features3(m, tsv_o, feature_set=dst, forward=True, kmer_length=6)
print features_2

tr, tr_l, xt, xt_l = collect_data_vectors2(1, aln, True, 0, 1.0, m, 100,
                                           feature_set=dst)
print tr


#d = cull_motif_features(747, tsv_t, True)
#print d
#d2 = cull_motif_features2(747, tsv_t)
#print d2
#print d2.groupby('ref_pos').head(2)
#d747 = d2.ix[d2['ref_pos'] == 747]
#events = list(chain(*d2.ix[d2['ref_pos'] == 747].drop('ref_pos', 1)[:5].values.tolist()))
#print len(events)
#r = d747.sort_values('posterior', ascending=False)
#print r.drop_duplicates(subset='delta_mean')
#r = r.drop('ref_pos', 1)
#l = r.values.tolist()









