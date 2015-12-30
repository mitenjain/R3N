#!/usr/bin/env python

from toy_datasets import *
from neural_network import *
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
train, test = load_digit_dataset(1000, 0.1)
X = np.array([x[0] for x in train])
Y = [y[1] for y in train]
Y = np.asarray(Y)
X2 = np.array([x[0] for x in test])
Y2 = [y[1] for y in test]

# Testing Neural Nets #
'''
net = mini_batch_sgd(train_data=X, labels=Y,
                     xTrain_data=X2, xTrain_labels=Y2,
                     learning_rate=0.001, L1_reg=0.0, L2_reg=0.00,
                     epochs=5000, batch_size=10, hidden_dim=[10, 10],
                     model_type="ReLUthreeLayer", model_file=None,
                     trained_model_dir="./testRun/")
'''

'''
t1 = timeit.Timer("cull_motif_features(747, '../marginAlign/cPecan/tests/temp/tempFiles_alignment/makeson_PC_MA_286_R7.3_ZYMO_C_1_09_11_15_1714_1_ch1_file1_strand.e.forward.tsv', True)",
                  setup="from utils import cull_motif_features")
s1 = t1.timeit(number=1)

t2 = timeit.Timer("cull_motif_features2(747, '../marginAlign/cPecan/tests/temp/tempFiles_alignment/makeson_PC_MA_286_R7.3_ZYMO_C_1_09_11_15_1714_1_ch1_file1_strand.e.forward.tsv', True)",
                  setup="from utils import cull_motif_features2")
s2 = t2.timeit(number=1)

print s1, s2, s1/s2
'''

tsv_t = "../marginAlign/cPecan/tests/temp/tempFiles_alignment/" \
        "makeson_PC_MA_286_R7.3_ZYMO_C_1_09_11_15_1714_1_ch1_file1_strand.vl.forward.tsv"

aln = "../marginAlign/cPecan/tests/oneFile/"

tr1, tr_l1, xt1, xt_l1 = collect_data_vectors(5, aln, True, 0, 1.0, 757, 100)
tr, tr_l, xt, xt_l = collect_data_vectors2(5, aln, True, 0, 1.0, 757, 100)

print tr, '\n', tr1

'''
t1 = timeit.Timer("collect_data_vectors2(1, aln, True, 0, 1.0, 757, 100)",
                  setup="from utils import collect_data_vectors2 \n"
                        "aln='../marginAlign/cPecan/tests/oneFile/'")
s1 = t1.timeit(number=2)
t2 = timeit.Timer("collect_data_vectors(1, aln, True, 0, 1.0, 757, 100)",
                  setup="from utils import collect_data_vectors \n"
                        "aln='../marginAlign/cPecan/tests/oneFile/'")
s2 = t2.timeit(number=2)
print s1
print s2
print s2/s1
'''
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









