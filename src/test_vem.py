#!/usr/bin/env python

import numpy as np
from variational_em import VariationalEM

model = VariationalEM(num_people=20, num_groups=2)

# y = np.loadtxt("../data/Y_alpha0.1_K2_N20.txt", dtype="float64", delimiter=",")

# Two obvious groups!!!!
y = np.zeros((20, 20))
y[0:10, 0:10] = 1
y[10:20, 10:20] = 1

model.train(y)

print "alpha:"
print model.alpha
print "b:"
print model.b
