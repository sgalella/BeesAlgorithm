import numpy as np

import optimization_functions as opt
from bees_algorithm import BeesAlgorithm
from visualization import VisualizeSearch


np.random.seed(1234)
limits = (-5, 5, -3, 3)  # x_min, x_max, y_min, y_max
resolution = 100
num_iterations = 20
landscape = opt.SphereLandscape(limits, resolution)
bees = BeesAlgorithm(landscape, n=20, m=3, e=1, nep=10, nsp=7, ngh=0.5)
VisualizeSearch.show_all(bees, num_iterations)
