import unittest

import bees_algorithm as ba


limits = [-5, 5, -3, 3]
resolution = 100
num_iterations = 100


def run_search(algorithm, num_iterations=100):
    for _ in range(num_iterations):
        recruiters_e, best_e, recruiters_m_e, best_m_e = algorithm.recruit_scouts()
        algorithm.abandon_locations(recruiters_e, best_e, recruiters_m_e, best_m_e)


class TestConvergence(unittest.TestCase):

    def test_sphere(self):
        landscape = ba.SphereLandscape(limits, resolution)
        bees = ba.BeesAlgorithm(landscape, n=20, m=3, e=1, nep=10, nsp=7, ngh=0.5)
        run_search(bees, num_iterations)
        self.assertAlmostEqual(bees.best_fitness, 1)

    def test_grickwank(self):
        landscape = ba.GrickwankLandscape(limits, resolution)
        bees = ba.BeesAlgorithm(landscape, n=20, m=3, e=1, nep=10, nsp=7, ngh=0.5)
        run_search(bees, num_iterations)
        self.assertAlmostEqual(bees.best_fitness, 1)

    def test_himmelblau(self):
        landscape = ba.HimmelblauLandscape(limits, resolution)
        bees = ba.BeesAlgorithm(landscape, n=20, m=3, e=1, nep=10, nsp=7, ngh=0.5)
        run_search(bees, num_iterations)
        self.assertAlmostEqual(bees.best_fitness, 1)

    def test_ackley(self):
        landscape = ba.AckleyLandscape(limits, resolution)
        bees = ba.BeesAlgorithm(landscape, n=20, m=3, e=1, nep=10, nsp=7, ngh=0.5)
        run_search(bees, num_iterations)
        self.assertAlmostEqual(bees.best_fitness, 1)

    def test_rastringin(self):
        landscape = ba.RastringinLandscape(limits, resolution)
        bees = ba.BeesAlgorithm(landscape, n=20, m=3, e=1, nep=10, nsp=7, ngh=0.5)
        run_search(bees, num_iterations)
        self.assertAlmostEqual(bees.best_fitness, 1)


if __name__ == '__main__':
    unittest.main()
