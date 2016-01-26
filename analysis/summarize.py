#!/usr/bin/env python

import numpy as np
import sys
import os

results = []

path = sys.argv[1]

result_files = [x for x in os.listdir(path) if x.endswith(".tsv")]

for f in result_files:
    for line in open(path + f, 'r'):
        if line.startswith(">"):
            line = line[1:]
            line = line.split()
            results.append((line[0], line[1]))

for result in results:
    print >> sys.stdout, "{0}\t{1:.2f}".format(result[0], (float(result[1]) * 100))

results = [float(x[1]) * 100 for x in results]
print "Avg: {0:.2f}".format(np.mean(results))
