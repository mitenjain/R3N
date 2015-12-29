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

#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
#d = mnist.train.next_batch(50)
#X = d[0]
#Y = d[1]

# testing library

#net = NeuralNetwork([2, 10, 2], hyperbolic_tangent)

'''
net = NeuralNetwork(input_dim=X.shape[1],
                    nb_classes=len(set(Y)),
                    hidden_dims=[10],
                    activation_function=hyperbolic_tangent)

net.mini_batch_sgd(training_data=X,
                   labels=Y,
                   epochs=1000,
                   batch_size=10,
                   epsilon=0.001,
                   lbda=0.001,
                   print_loss=True)
'''

#net.fit(X, Y, epochs=5000, epsilon=0.001, lbda=0.001, print_loss=True)
#t = net.evaluate(X2, Y2)
#print net.predict_old(X2)[1:10]
#print net.predict(X2)[1:10]
#print net.predict_old(X2)[1:10] == net.predict(X2)[1:10]
#plot_decision_boundary(lambda x: np.argmax(net.predict(x), axis=1),
#                       X, Y)

#dataset = load_data("../neural-networks-and-deep-learning/data/mnist.pkl.gz")

#X, Y = dataset[0]
#X2, Y2 = dataset[1]
#test_set_x, test_set_y = dataset[2]
'''
# Testing Neural Nets #
net = mini_batch_sgd(train_data=X, labels=Y,
                     xTrain_data=X2, xTrain_labels=Y2,
                     learning_rate=0.01, L1_reg=0.0, L2_reg=0.001,
                     epochs=5000, batch_size=20, hidden_dim=[100, 100],
                     model_type="ReLUthreeLayer", model_file=None,
                     trained_model_dir="./testRun/")
'''

'''
#data = np.loadtxt(tsv_t, dtype=str, usecols=(0, 1, 5, 6, 8, 9, 10))
data = pd.read_table(tsv_t, usecols=(0, 1, 4, 5, 6, 8, 9, 10), dtype={'ref_pos': np.int32,
                                                                      'event_idx': np.int32,
                                                                      'strand': np.str,
                                                                      'event_mean': np.float64,
                                                                      'event_noise': np.float64,
                                                                      'prob': np.float64,
                                                                      'E_mean': np.float64,
                                                                      'E_noise': np.float64},
                     header=None, names=['ref_pos', 'event_idx', 'strand', 'event_mean', 'event_noise', 'prob', 'E_mean', 'E_noise'])

cytosine_motifs = [747, 354, 148, 796, 289, 363, 755, 626, 813, 653, 525, 80, 874]

motif_range = range(747, 747 + 6)

#motif_rows = data[data['ref_pos'].isin(motif_range)]
motif_rows = data.ix[(data['ref_pos'].isin(motif_range)) & (data['strand'] == 't')]

print motif_rows
features = pd.DataFrame({"delta_mean": motif_rows['event_mean'] - motif_rows['E_mean'],
                         "posterior": motif_rows['prob']})
print motif_rows['event_mean'] - motif_rows['E_mean'], motif_rows['prob']
print features
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

tr1, tr_l1, xt1, xt_l1 = collect_data_vectors(1, aln, True, 0, 1.0, 757, 100)
tr, tr_l, xt, xt_l = collect_data_vectors2(1, aln, True, 0, 1.0, 757, 100)

print tr, '\n', tr1

'''
t1 = timeit.Timer("collect_data_vectors2(2, aln, True, 0, 1.0, 757, 100)",
                  setup="from utils import collect_data_vectors2 \n"
                        "aln='../marginAlign/cPecan/tests/oneFile/'")
s1 = t1.timeit(number=2)
t2 = timeit.Timer("collect_data_vectors(2, aln, True, 0, 1.0, 757, 100)",
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









