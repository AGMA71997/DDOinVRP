import numpy
import random
import math

class Instance_Generator(object):

    def __init__(self, N):
        self.time_matrix = self.create_time_matrix(N)
        minimum_margin=math.ceil(self.time_matrix.max())
        self.time_windows = self.create_time_windows(N,minimum_margin)
        self.vehicle_capacity = 15
        self.demands = self.create_demands(N)
        self.time_limit = 8

    def create_time_matrix(self, N):

        time_matrix = numpy.zeros((N + 1, N + 1))
        customer_locations = numpy.random.random_sample((N + 1, 2))
        customer_locations[0, 0] = 0
        customer_locations[0, 1] = 0

        for i in range(N + 1):
            for j in range(N + 1):
                if i != j:
                    time_matrix[i, j] = numpy.linalg.norm(customer_locations[i, :] - customer_locations[j, :])

        return time_matrix * 2

    def create_time_windows(self, N, minimum_margin):
        time_windows = numpy.zeros((N + 1, 2))
        for i in range(N + 1):
            if i != 0:
                time_windows[i, 0] = random.randint(0, 10)
                time_windows[i, 1] = random.randint(time_windows[i, 0] + minimum_margin, 18)

        return time_windows

    def create_demands(self, N):
        demands = numpy.zeros(N + 1)
        for i in range(N + 1):
            if i != 0:
                demands[i] = random.randint(1, 3)

        return demands


def main():
    VRP_instance = Instance_Generator(50)


if __name__ == "__main__":
    main()
