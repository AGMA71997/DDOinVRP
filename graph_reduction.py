import numpy
import math
from utils import create_price


class Arc_Reduction(object):

    def __init__(self, prices, duals):
        self.prices = numpy.copy(prices)
        numpy.fill_diagonal(self.prices, math.inf)
        self.duals = duals.copy()
        self.N = len(self.prices)

    def BE1(self, time_matrix, alpha=0.5):
        threshold = alpha * max(self.duals)
        self.prices[time_matrix > threshold] = math.inf
        return self.prices

    def BE2(self, alpha=0.25):
        edge_count = math.ceil(alpha * self.N ** 2)
        indices = self.prices.ravel().argsort()

        relevant_prices = self.prices.ravel()[indices[0:edge_count]]

        for x in range(self.N):
            for y in range(1, self.N):
                if self.prices[x, y] not in relevant_prices:
                    self.prices[x, y] = math.inf
        return self.prices

    def BE3(self, alpha=0.5):
        M = math.ceil((self.N - 2)*alpha)

        for i in range(self.N):
            cus_prices = numpy.sort(self.prices[i, :])
            cus_prices = cus_prices[0:M]
            for j in range(1, self.N):
                if self.prices[i, j] not in cus_prices:
                    self.prices[i, j] = math.inf

        return self.prices

    def BN(self, beta=0.9):

        probs = {}
        max_dual = max(self.duals)
        min_dual = min(self.duals[1:])
        for x in range(1, self.N):
            probs[x] = (self.duals[x] - min_dual) / (max_dual - min_dual) * beta

        for x in range(self.N):
            for y in range(1, self.N):
                random_var = numpy.random.random()
                if random_var > probs[y]:
                    self.prices[x, y] = math.inf

        return self.prices

    def BP(self):
        pass

    def euclidean_diff_to_dual_ratio(self, time_matrix):
        for i in range(self.N):
            for j in range(1, self.N):
                if self.prices[i, j] > 0:
                    arc_mute = True
                    for k in range(self.N):
                        time_diff = time_matrix[i, j] + time_matrix[j, k] - time_matrix[i, k]
                        if time_diff < self.duals[j]:
                            arc_mute = False
                            break

                    if arc_mute:
                        self.prices[i, j] = math.inf

        return self.prices


class Node_Reduction(object):
    def __init__(self, coords, duals=None):
        if duals is not None:
            self.duals = duals.copy()
        self.coords = numpy.copy(coords)
        self.N = len(self.coords)

    def dual_based_elimination(self):
        for x in range(1, self.N):
            if self.duals[x] == 0:
                self.coords[x, :] = math.inf
        return self.coords

    def price_based_elimination(self, time_matrix):
        time_matrix_2 = numpy.copy(time_matrix)
        numpy.fill_diagonal(time_matrix_2, math.inf)
        for x in range(1, self.N):
            smallest_tt = numpy.min(time_matrix_2[:, x])
            if self.duals[x] < smallest_tt:
                self.coords[x, :] = math.inf
        return self.coords

    def reduce_by_indices(self,indices):
        for x in range(1, self.N):
            if x - 1 not in indices:
                self.coords[x, :] = math.inf

        return self.coords




def main():
    num_customers = 30
    coords = numpy.random.random((num_customers + 1, 2))
    coords[0, :] = numpy.array([0, 0])
    print(coords)
    time_matrix = numpy.random.uniform(low=0.25, high=2.5, size=(num_customers + 1, num_customers + 1))
    duals = list(numpy.random.uniform(low=0, high=10, size=num_customers + 1))
    duals[0] = 0
    print(duals)

    prices = create_price(time_matrix, duals) * -1
    # print(prices)

    AC = Arc_Reduction(prices, duals)
    reduced_prices = AC.BN()
    # print(reduced_prices)

    NC = Node_Reduction(duals, coords)
    reduced_nodes = NC.price_based_elimination(time_matrix)
    print(reduced_nodes)


if __name__ == "__main__":
    main()
