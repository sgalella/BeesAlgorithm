from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial


class FitnessLandscape(ABC):
    """ Template for building landscapes. """
    def __init__(self, limits, resolution):
        """
        Initialize bounds and fitness function.

        Args:
            limits (list): Boundaries of the landscape: [x_min, x_max, y_min, y_max]
            resolution (int): Number of points per dimension.
        """
        self.limits = limits
        self.resolution = resolution
        self.X, self.Y = self._create_meshgrid()
        self.coords, self.tree = self._generate_coords()
        self.fitness_function = self._calculate_fitness().reshape(self.resolution, self.resolution)
        self.max, self.min = np.max(self.fitness_function), np.min(self.fitness_function)

    def _generate_coords(self):
        """
        Generates array of coordinates and tree for positions interpolation.

        Returns:
            tuple: Coordinates and tree for position lookup.
        """
        coords = np.dstack([self.X.ravel(), self.Y.ravel()])[0]
        return coords, spatial.cKDTree(coords)

    def _create_meshgrid(self):
        """
        Builds up the grid of the landscape. Each point corresponds to a coordinate in the space.

        Returns:
            tuple: Arrays containing coordinates of the meshgrid.
        """
        x = np.linspace(self.limits[0], self.limits[1], self.resolution)
        y = np.linspace(self.limits[2], self.limits[3], self.resolution)
        X, Y = np.meshgrid(x, y)
        return X, Y

    def evaluate_fitness(self, pos):
        """
        Computes the fitness of individual at position given the fitness function of the landscape.

        Args:
            pos (tuple): x and y of the individual

        Returns:
            float: Normalized fitness of the individual. Individuals in the minima will have a fitness close to 1.
        """
        _, index = self.tree.query(pos)
        return 1 - (self.fitness_function[index // self.resolution][index % self.resolution] - self.min) / (self.max - self.min)

    def plot(self):
        """ Displays the landscape using contour maps. """
        cs = plt.contour(self.X, self.Y, self.fitness_function)
        plt.clabel(cs, inline=1, fontsize=6)
        plt.imshow(self.fitness_function, extent=self.limits, origin="lower", alpha=0.3)
        plt.xlim([self.limits[0], self.limits[1]])
        plt.ylim([self.limits[2], self.limits[3]])

    @abstractmethod
    def _calculate_fitness(self):
        """ Creates the fitness landscape given a function.
            Check https://en.wikipedia.org/wiki/Test_functions_for_optimization?wprov=srpw1_0
            for more information.
        """
        pass


class SphereLandscape(FitnessLandscape):
    def _calculate_fitness(self):
        return self.X ** 2 + self.Y ** 2


class GrickwankLandscape(FitnessLandscape):
    def _calculate_fitness(self):
        return 1 + (self.X ** 2 / 4000) + (self.Y ** 2 / 4000) - np.cos(self.X / np.sqrt(2)) - np.cos(self.Y / np.sqrt(2))


class HimmelblauLandscape(FitnessLandscape):
    def _calculate_fitness(self):
        return (self.X ** 2 + self.Y - 11) ** 2 + (self.X + self.Y ** 2 - 7) ** 2


class AckleyLandscape(FitnessLandscape):
    def _calculate_fitness(self):
        return (-20 * np.exp(-0.2 * np.sqrt(0.5 * (self.X ** 2 + self.Y ** 2))) - np.exp(0.5 * np.cos(2 * np.pi * self.X)
                + np.cos(2 * np.pi * self.Y)) + np.exp(1) + 20)


class RastringinLandscape(FitnessLandscape):
    def _calculate_fitness(self):
        return 20 + self.X ** 2 - 10 * np.cos(2 * np.pi * self.X) - 10 * np.cos(2 * np.pi * self.Y)
