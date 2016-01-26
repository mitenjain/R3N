#!/usr/bin/env python
import sys
sys.path.append("../")
from lib.utils import cull_motif_features4, collect_data_vectors2

tsv_t = "../../marginAlign/cPecan/tests/temp/tempFiles_alignment/" \
        "makeson_PC_MA_286_R7.3_ZYMO_C_1_09_11_15_1714_1_ch1_file1_strand.vl.forward.tsv"

aln = "../marginAlign/cPecan/tests/temp/tempFiles_alignment/*.tsv"

aln2 = "../marginAlign/cPecan/tests/test_alignments/newf_conditional_model/C/tempFiles_alignment/*.forward.tsv"

m = [300, 747]
dst = "all"
strand = "t"

features = cull_motif_features4(m, tsv_t, strand, feature_set=dst, kmer_length=6)
print features.ix[features['ref_pos'] == 300]
print features.ix[features['ref_pos'] == 300]['delta_mean']

#events = []
#for strand in ["t", "c"]:
#    events += list(chain(
#                         *features.ix[(features['ref_pos'] == 300) & (features['strand'] == strand)]
#                         .drop(['ref_pos', 'strand'], 1)[:1].values.tolist()))

'''
tr, xtr, ts = collect_data_vectors2(events_per_pos=1,
                                    label=0,
                                    portion=0.5,
                                    files=aln,
                                    strand=strand,
                                    motif_starts=m,
                                    dataset_title="test",
                                    max_samples=10,
                                    feature_set=dst)

print tr
'''