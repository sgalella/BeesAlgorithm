import matplotlib.pyplot as plt
import numpy as np


class BeesAlgorithm:
    """ Bees algorithm optimization """
    def __init__(self, landscape, n, m, e, nep, nsp, ngh):
        """
        Initializes the algorithm.

        Args:
            landscape (Landscape): Fitness plane where bees interact.
            n (int): Number of scout bees.
            m (int): Number of sites selected (out of n).
            e (int): Number of best sites (out of m).
            nep (int): Number of bees recruited e.
            nsp (int): Number of bees recruited m.
            ngh (int): Initial size patches.
        """
        self.landscape = landscape
        self.n = n
        self.m = m
        self.e = e
        self.nep = nep
        self.nsp = nsp
        self.ngh = ngh
        self.best_fitness = None
        self.positions = np.zeros((self.n, 2))
        self.fitness = np.zeros((self.n, ))
        self._initialize_position()
        self._calculate_fitness()

    def _initialize_position(self):
        """ Initializes position of bee randomly in landscape. """
        min_x, max_x, min_y, max_y = self.landscape.limits
        self.positions[:, 0] = np.random.uniform(min_x, max_x, size=(self.n, ))
        self.positions[:, 1] = np.random.uniform(min_y, max_y, size=(self.n, ))

    def _calculate_fitness(self):
        """ Computes the fitness for the bees. """
        for idx in range(self.n):
            pos = self.positions[idx, :]
            self.fitness[idx] = self.landscape.evaluate_fitness(pos)
        self.best_fitness = max(self.fitness)

    def update_positions(self, recruiters, best):
        """
        Updates position of bees.

        Args:
            recruiters (np.array): Recruiters bees.
            best (np.array): Best bees.
        """
        min_x, max_x, min_y, max_y = self.landscape.limits
        assert len(recruiters) > len(best), "Number of scouts is lesser than number of optimal solutions."
        ratio_recruiter = len(recruiters) // len(best)
        for current_num, current_pos in enumerate(best):
            current_recruiters = (recruiters[current_num * ratio_recruiter: (current_num + 1) * ratio_recruiter]
                                  if (current_num + 1) * ratio_recruiter <= len(best) else recruiters[current_num * ratio_recruiter:])
            for idx in current_recruiters:
                self.positions[idx, :] = self.positions[current_pos, :] + 2 * self.ngh * np.random.random(size=(1, 2)) - self.ngh
        self._calculate_fitness()

    def recruit_scouts(self):
        """ Recruits scout bees accordint to parameters. """
        best_fitness_all = self.fitness.argsort()
        recruiters_e = best_fitness_all[:self.nep]  # Recruiters in best positions
        recruiters_m_e = best_fitness_all[self.nep:self.nsp + self.nep]  # Recruiters in m - e positions
        best_e = best_fitness_all[self.nsp + self.nep + self.m - self.e:]
        best_m_e = best_fitness_all[self.nsp + self.nep: self.nsp + self.nep + self.m - self.e]
        self.update_positions(recruiters_e, best_e)
        self.update_positions(recruiters_m_e, best_m_e)
        return (recruiters_e, best_e, recruiters_m_e, best_m_e)

    def abandon_locations(self, recruiters_e, best_e, recruiters_m_e, best_m_e):
        """
        Relocates bees again in the landscape (except best).

        Args:
            recruiters_e (np.array): Recruiters on best bees.
            best_e (np.array): Best bees in landscape.
            recruiters_m_e (np.array): Recruiters on rest of the best bees.
            best_m_e (np.array): Rest of bees bees in the landscape.
        """
        min_x, max_x, min_y, max_y = self.landscape.limits
        # Abandon best places leaving a single scout
        patch_e = np.concatenate([recruiters_e, best_e])
        best_fitness_patch_e = self.fitness[patch_e].argsort()[::-1]
        self.positions[patch_e[best_fitness_patch_e][1:], 0] = np.random.uniform(min_x, max_x, size=(len(best_fitness_patch_e[1:]), ))
        self.positions[patch_e[best_fitness_patch_e][1:], 1] = np.random.uniform(min_y, max_y, size=(len(best_fitness_patch_e[1:]), ))
        # Abandon rest also leaving a single scout
        patch_m_e = np.concatenate([recruiters_m_e, best_m_e])
        best_fitness_patch_m_e = self.fitness[patch_m_e].argsort()[::-1]
        self.positions[patch_m_e[best_fitness_patch_m_e][1:], 0] = np.random.uniform(min_x, max_x, size=(len(best_fitness_patch_m_e[1:]), ))
        self.positions[patch_m_e[best_fitness_patch_m_e][1:], 1] = np.random.uniform(min_y, max_y, size=(len(best_fitness_patch_m_e[1:]), ))
        self._calculate_fitness()

    def plot(self):
        """ Plots bees in landscape. """
        best_fitness_idx = self.fitness.argsort()[::-1][:self.e]
        for idx in range(self.n):
            if idx in best_fitness_idx:
                circle = plt.Circle(self.positions[idx, :], radius=self.ngh, color="gold", alpha=0.5)
                plt.gca().add_patch(circle)
                plt.plot(self.positions[idx, 0], self.positions[idx, 1], "darkorange", marker="*")
            else:
                circle = plt.Circle(self.positions[idx, :], radius=self.ngh, color="r", alpha=0.1)
                plt.gca().add_patch(circle)
                plt.plot(self.positions[idx, 0], self.positions[idx, 1], "r*")
        plt.title(f"Best fitness: {self.best_fitness:.2f}")
