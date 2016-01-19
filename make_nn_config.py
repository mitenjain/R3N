#!/usr/bin/env python

import cPickle
import os
import numpy as np


class TargetRegions(object):
    def __init__(self, tsv, already_sorted=False):
        assert(os.stat(tsv).st_size != 0), "Empty regions file"

        self.region_array = np.loadtxt(tsv,
                                       usecols=(0, 1),
                                       dtype=np.int32)

        if len(self.region_array.shape) == 1:
            a = np.empty([1, 2], dtype=np.int32)
            a[0] = self.region_array
            self.region_array = a

        if not already_sorted:
            self.region_array = np.sort(self.region_array, axis=1)

###############################################################################
j = {
    "experiment_name": "individual cytosine motif classification",
    "hidden_dim": [50, 10],
    "model_type": "ReLUthreeLayer",
    "sites": []
}
for m in [747, 354, 148, 796, 289, 363, 755, 626, 813, 653, 525, 80, 874]:
    m_list = [m]
    d = dict()
    d['motif_start_position'] = [m_list, m_list, m_list]
    d['title'] = str(m)
    j['sites'].append(d)
cPickle.dump(j, open("./configs/indivCytosineZymo.pkl", 'w'))
###############################################################################
j = {
    "experiment_name": "individual cytosine motif classification 4 Layer",
    "hidden_dim": [50, 100, 50],
    "model_type": "fourLayer",
    "sites": []
}
for m in [747, 354, 148, 796, 289, 363, 755, 626, 813, 653, 525, 80, 874]:
    m_list = [m]
    d = dict()
    d['motif_start_position'] = [m_list, m_list, m_list]
    d['title'] = str(m)
    j['sites'].append(d)
cPickle.dump(j, open("./configs/indivCytosineZymo_4L.pkl", 'w'))
###############################################################################
j = {
    "experiment_name": "individual cytosine motif classification ConvNet",
    "hidden_dim": 100,
    "model_type": "ConvNet3",
    "extra_args": {
        "batch_size": 5,
        "n_filters": [10],
        "n_channels": [1],
        "data_shape": [3, 6],
        "filter_shape": [1, 3],
        "poolsize": (2, 2),
    },
    "sites": []
}
for m in [747, 354, 148, 796, 289, 363, 755, 626, 813, 653, 525, 80, 874]:
    m = [m]
    d = dict()
    d['motif_start_position'] = [m, m, m]
    d['title'] = str(m)
    j['sites'].append(d)
cPickle.dump(j, open("./configs/indivCytosineZymo_conv.pkl", 'w'))
###############################################################################
j = {
    "experiment_name": "indvidual null motif classification",
    "hidden_dim": [100, 100],
    "model_type": "ReLUthreeLayer",
    "sites": []
}
for m in [11, 62, 87, 218, 295, 371, 383, 457, 518, 740, 785, 805, 842, 866]:
    m_list = [m]
    d = dict()
    d['motif_start_position'] = [m_list, m_list, m_list]
    d['title'] = str(m)
    j['sites'].append(d)
cPickle.dump(j, open("./configs/indivNullZymo.pkl", 'w'))
###############################################################################
j = {
    "experiment_name": "All cytosine motifs classification",
    "hidden_dim": 100,
    "model_type": "ConvNet3",
    "extra_args": {
        "batch_size": 10,
        "n_filters": [10],
        "data_shape": [3, 6],
        "filter_shape": [1, 3],
        "poolsize": (2, 2),
    },
    "sites": []
}
for m in [[747, 354, 148, 796, 289, 363, 755, 626, 813, 653, 525, 80, 874]]:
    d = dict()
    d['motif_start_position'] = [m, m, m]
    d['title'] = "all_zymo_cytosine"
    j['sites'].append(d)
cPickle.dump(j, open("./configs/all_zymo_cytosine.pkl", 'w'))
###############################################################################
j = {
    "experiment_name": "all null motifs classification",
    "hidden_dim": [100, 100],
    "model_type": "ReLUthreeLayer",
    "sites": []
}
for m in [[11, 62, 87, 218, 295, 371, 383, 457, 518, 740, 785, 805, 842, 866]]:
    d = dict()
    d['motif_start_position'] = [m, m, m]
    d['title'] = "all_zymo_null"
    j['sites'].append(d)
cPickle.dump(j, open("./configs/all_zymo_null.pkl", 'w'))
###############################################################################
ecoli_positive_A = TargetRegions("./regions/ecoli_positive_A.tsv")
ecoli_positive_T = TargetRegions("./regions/ecoli_positive_T.tsv")
ecoli_negative_A = TargetRegions("./regions/ecoli_null_A.tsv")
ecoli_negative_T = TargetRegions("./regions/ecoli_null_T.tsv")

pos_A_list = [x[0] for x in ecoli_positive_A.region_array]
pos_T_list = [x[0] for x in ecoli_positive_T.region_array]
neg_A_list = [x[0] for x in ecoli_negative_A.region_array]
neg_T_list = [x[0] for x in ecoli_negative_T.region_array]


j = {
    "experiment_name": "ecoli CCAGG",
    "hidden_dim": [500, 500],
    "model_type": "ReLUthreeLayer",
    "sites": []
}
d = dict()
d['motif_start_position'] = [pos_A_list, neg_A_list]
d['title'] = "ecoli_CCAGG"
j['sites'].append(d)
cPickle.dump(j, open("./configs/ecoli_CCAGG.pkl", 'w'))
###############################################################################
j = {
    "experiment_name": "ecoli CCTGG",
    "hidden_dim": [100, 100],
    "model_type": "ReLUthreeLayer",
    "sites": []
}
d = dict()
d['motif_start_position'] = [pos_T_list, neg_T_list]
d['title'] = "ecoli_CCTGG"
j['sites'].append(d)
cPickle.dump(j, open("./configs/ecoli_CCTGG.pkl", 'w'))
###############################################################################
ecoli_pos_intersect_A = TargetRegions("./regions/ecoli_positive_intersect_A.tsv")
ecoli_neg_intersect_A = TargetRegions("./regions/ecoli_negative_intersect_A.tsv")
pos_int_A_list = [x[0] for x in ecoli_pos_intersect_A.region_array]
neg_int_A_list = [x[0] for x in ecoli_neg_intersect_A.region_array]

j = {
    "experiment_name": "ecoli CCAGG - intersect",
    "hidden_dim": [50, 10],
    "model_type": "ReLUthreeLayer",
    "sites": []
}
d = dict()
d['motif_start_position'] = [pos_int_A_list, neg_int_A_list]
d['title'] = "ecoli_CCAGG_intersection"
j['sites'].append(d)
cPickle.dump(j, open("./configs/ecoli_intersect_CCAGG.pkl", 'w'))
###############################################################################
