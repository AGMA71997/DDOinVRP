import numpy
import random

class Instance_Generator(object):

    def __init__(self, N):
        self.distance_matrix=self.create_distance_matrix(N)
        self.time_windows=self.create_time_windows(N)
        print(self.distance_matrix)
        print(self.time_windows)


    def create_distance_matrix(self, N):

        distance_matrix = numpy.zeros((N + 1, N + 1))
        customer_locations = numpy.random.random_sample((N + 1, 2))
        customer_locations[0, 0] = 0
        customer_locations[0, 1] = 0

        for i in range(N + 1):
            for j in range(N + 1):
                if i != j:
                    distance_matrix[i, j] = numpy.linalg.norm(customer_locations[i, :] - customer_locations[j, :])

        return distance_matrix*2

    def create_time_windows(self, N):

        time_windows = numpy.zeros((N+1 , 2))
        for i in range(N):
            if i!=0:
                time_windows[i, 0] = random.randint(6,16)
                time_windows[i, 1] = random.randint(time_windows[i, 0]+2,24)

        return time_windows

def main():
    IR=Instance_Generator(50)

if __name__ == "__main__":
    main()