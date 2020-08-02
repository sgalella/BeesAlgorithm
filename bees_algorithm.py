import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1234)

MAX_ITERATIONS = 20


class FunctionNotFoundError(Exception):
    def __init__(self, name):
        super().__init__(f"'{name}' optimization function not exist.")


class Landscape:
    def __init__(self, limits, resolution, func_name="Sphere"):
        self.limits = (limits[0], limits[1], limits[2], limits[3])
        self.X, self.Y = self.__create_meshgrid(resolution)
        self.func = self.__get_func(func_name.lower())

    def __get_func(self, func_name):
        if func_name == "sphere":
            return self.X ** 2 + self.Y ** 2
        elif func_name == "gricwank":
            return 1 + (self.X ** 2 / 4000) + (self.Y ** 2 / 4000) - np.cos(self.X / np.sqrt(2)) - np.cos(self.Y / np.sqrt(2))
        elif func_name == "himmelblau":
            return (self.X ** 2 + self.Y - 11) ** 2 + (self.X + self.Y ** 2 - 7) ** 2
        elif func_name == "ackley":
            return (-20 * np.exp(-0.2 * np.sqrt(0.5 * (self.X ** 2 + self.Y ** 2))) - np.exp(0.5 * np.cos(2 * np.pi * self.X)
                    + np.cos(2 * np.pi * self.Y)) + np.exp(1) + 20)
        elif func_name == "rastringin":
            return 20 + self.X ** 2 - 10 * np.cos(2 * np.pi * self.X) - 10 * np.cos(2 * np.pi * self.Y)
        else:
            raise FunctionNotFoundError(func_name)

    def __create_meshgrid(self, resolution):
        x = np.linspace(self.limits[0], self.limits[1], resolution)
        y = np.linspace(self.limits[2], self.limits[3], resolution)
        X, Y = np.meshgrid(x, y)
        return X, Y

    def get_value_func(self, pos):
        pos_x, pos_y = pos
        _, j = np.unravel_index((np.abs(self.X - pos_x)).argmin(), self.func.shape)
        i, _ = np.unravel_index((np.abs(self.Y - pos_y)).argmin(), self.func.shape)
        return np.fabs(self.func[i, j] - np.max(self.func))

    def plot(self):
        cs = plt.contour(self.X, self.Y, self.func)
        plt.clabel(cs, inline=1, fontsize=6)
        plt.imshow(self.func, extent=self.limits, origin="lower", alpha=0.3)


class BeesAlgorithm:
    def __init__(self, landscape, n, m, e, nep, nsp, ngh):
        self.landscape = landscape
        self.n = n  # No. scout bees
        self.m = m  # No. sites selected (out of n)
        self.e = e  # No. best sites (out of m)
        self.nep = nep  # No. bees rectruited e
        self.nsp = nsp  # No. bees recruited m
        self.ngh = ngh  # Initial size patches
        self.positions = np.zeros((self.n, 2))
        self.fitness = np.zeros((self.n, ))
        self.__initialize_position()
        self.__calculate_fitness()

    def __initialize_position(self):
        min_x, max_x, min_y, max_y = self.landscape.limits
        self.positions[:, 0] = np.random.uniform(min_x, max_x, size=(self.n, ))
        self.positions[:, 1] = np.random.uniform(min_y, max_y, size=(self.n, ))

    def __calculate_fitness(self):
        for idx in range(self.n):
            pos = self.positions[idx, :]
            self.fitness[idx] = self.landscape.get_value_func(pos)

    def update_positions(self, recruiters, best):
        min_x, max_x, min_y, max_y = self.landscape.limits
        assert len(recruiters) > len(best), "Number of scouts is lesser than number of optimal solutions."
        ratio_recruiter = len(recruiters) // len(best)
        for current_num, current_pos in enumerate(best):
            current_recruiters = (recruiters[current_num * ratio_recruiter: (current_num + 1) * ratio_recruiter]
                                  if (current_num + 1) * ratio_recruiter <= len(best) else recruiters[current_num * ratio_recruiter:])
            for idx in current_recruiters:
                self.positions[idx, :] = self.positions[current_pos, :] + 2 * self.ngh * np.random.random(size=(1, 2)) - self.ngh
        self.__calculate_fitness()

    def recruit_scouts(self):
        best_fitness_all = self.fitness.argsort()
        recruiters_e = best_fitness_all[:self.nep]  # Recruiters in best positions
        recruiters_m_e = best_fitness_all[self.nep:self.nsp + self.nep]  # Recruiters in m - e positions
        best_e = best_fitness_all[self.nsp + self.nep + self.m - self.e:]
        best_m_e = best_fitness_all[self.nsp + self.nep: self.nsp + self.nep + self.m - self.e]
        self.update_positions(recruiters_e, best_e)
        self.update_positions(recruiters_m_e, best_m_e)
        return (recruiters_e, best_e, recruiters_m_e, best_m_e)

    def abandon_locations(self, recruiters_e, best_e, recruiters_m_e, best_m_e):
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
        self.__calculate_fitness()

    def plot(self):
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


def update_plots(limits, landscape, bees, colorbar=False):
    plt.cla()
    ax = plt.gca()
    ax.set_xlim([limits[0], limits[1]])
    ax.set_ylim([limits[2], limits[3]])
    landscape.plot()
    bees.plot()
    plt.title(f"Best fitness: {max(bees.fitness):.2f}")
    if colorbar:
        plt.colorbar(shrink=0.75)
    plt.draw()
    plt.pause(0.2)


def main():
    plt.figure(figsize=(8, 5))
    limits = (-5, 5, -3, 3)  # min_x, max_x, min_y, max_y
    landscape = Landscape(limits, 100)
    bees = BeesAlgorithm(landscape, 20, 3, 1, 10, 7, 0.5)
    update_plots(limits, landscape, bees, True)
    plt.ion()
    for _ in range(MAX_ITERATIONS):
        # Recruit scouts to best positions
        recruiters_e, best_e, recruiters_m_e, best_m_e = bees.recruit_scouts()
        update_plots(limits, landscape, bees)
        # Abandon positions and search for new potential places
        bees.abandon_locations(recruiters_e, best_e, recruiters_m_e, best_m_e)
        update_plots(limits, landscape, bees)
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
