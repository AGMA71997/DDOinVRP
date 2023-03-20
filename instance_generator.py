import numpy
import random
import math


class Instance_Generator(object):

    def __init__(self, N):
        random.seed(5)
        numpy.random.seed(25)
        self.N = N
        self.time_matrix = self.create_time_matrix()
        max_distance = math.ceil(self.time_matrix.max())
        max_distance_to_depot = math.ceil(self.time_matrix[:, 0].max())
        self.time_windows = self.create_time_windows(max_distance, max_distance_to_depot)
        self.vehicle_capacity = 10
        self.demands = self.create_demands()
        self.time_limit = 6.5
        self.service_times = self.create_service_times()

    def create_time_matrix(self):

        time_matrix = numpy.zeros((self.N + 1, self.N + 1))
        customer_locations = numpy.random.random_sample((self.N + 1, 2))
        customer_locations[0, 0] = 0
        customer_locations[0, 1] = 0

        for i in range(self.N + 1):
            for j in range(self.N + 1):
                if i != j:
                    time_matrix[i, j] = numpy.linalg.norm(customer_locations[i, :] - customer_locations[j, :])

        return time_matrix * 2

    def create_time_windows(self, minimum_margin,depot_increment):
        time_windows = numpy.zeros((self.N + 1, 2))
        for i in range(self.N + 1):
            if i != 0:
                time_windows[i, 0] = random.randint(0, 10)
                time_windows[i, 1] = random.randint(time_windows[i, 0] + minimum_margin, 18)

        time_windows[0,1]=18+depot_increment

        return time_windows

    def create_demands(self):
        demands = numpy.zeros(self.N + 1)
        for i in range(self.N + 1):
            if i != 0:
                demands[i] = random.randint(1, 3)

        return demands

    def create_service_times(self):
        service_times = numpy.zeros(self.N + 1)
        for i in range(self.N + 1):
            if i != 0:
                service_times[i] = random.uniform(0.2, 0.5)

        return service_times


def main():
    VRP_instance = Instance_Generator(50)


if __name__ == "__main__":
    main()
