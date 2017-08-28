import numpy as np
from variational_em import VariationalEM

y = np.loadtxt("data/Y_alpha0.1_K5_N20.txt", delimiter=',')
model = VariationalEM(num_people = 20, num_groups = 5, num_iterations = 100)

model.estimate(y)
