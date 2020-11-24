import matplotlib.pyplot as plt


class VisualizeSearch:
    @staticmethod
    def show_all(algorithm, num_iterations=100):
        """
        Shows the evolution of the solutions in the landscape.

        Args:
            algorithm (CuckooSearch): Cuckoo search initialized.
            num_iterations (int): Number of iterations to run the algorithm.
        """
        plt.figure(figsize=(8, 5))
        algorithm.plot()
        algorithm.landscape.plot()
        plt.colorbar(shrink=0.75)
        plt.ion()
        for _ in range(num_iterations):
            # Recruit scouts to best positions
            recruiters_e, best_e, recruiters_m_e, best_m_e = algorithm.recruit_scouts()
            plt.cla()
            algorithm.plot()
            algorithm.landscape.plot()
            plt.draw()
            plt.pause(0.1)

            # Abandon positions and search for new potential places
            algorithm.abandon_locations(recruiters_e, best_e, recruiters_m_e, best_m_e)
            plt.cla()
            algorithm.plot()
            algorithm.landscape.plot()
            plt.draw()
            plt.pause(0.1)
        recruiters_e, best_e, recruiters_m_e, best_m_e = algorithm.recruit_scouts()
        plt.cla()
        algorithm.plot()
        algorithm.landscape.plot()
        plt.ioff()
        plt.show()

    @staticmethod
    def show_last(algorithm, num_iterations=100):
        """
        Shows the last evolution of the solutions in the landscape.

        Args:
            algorithm (CuckooSearch): Cuckoo search initialized.
            num_iterations (int): Number of iterations to run the algorithm.
        """
        for _ in range(num_iterations):
            recruiters_e, best_e, recruiters_m_e, best_m_e = algorithm.recruit_scouts()
            algorithm.abandon_locations(recruiters_e, best_e, recruiters_m_e, best_m_e)
        recruiters_e, best_e, recruiters_m_e, best_m_e = algorithm.recruit_scouts()
        plt.figure(figsize=(8, 5))
        algorithm.landscape.plot()
        algorithm.plot()
        plt.colorbar(shrink=0.75)
        plt.show()
