import numpy as np
import gurobipy as gb
from instance_generator import Instance_Generator
import math


def solve_vrp_with_time_windows(vehicle_capacity, time_matrix, demands, time_windows, time_limit, num_customers):
    # Ensure all input lists are of the same length
    assert len(time_matrix) == len(demands) == len(time_windows)

    initial_routes, costs, ordered_routes = initialize_columns(num_customers, time_matrix)

    # Initialize the master problem
    master_problem = MasterProblem(num_customers, initial_routes, costs, ordered_routes)

    # Iterate until optimality is reached
    while True:
        master_problem.solve()
        duals = master_problem.retain_duals()
        subproblem = Subproblem(num_customers, vehicle_capacity, time_matrix, demands, time_windows, time_limit,
                                duals)
        ordered_route, reduced_cost = subproblem.solve()
        cost = sum(time_matrix[ordered_route[i], ordered_route[i + 1]] for i in range(len(ordered_route) - 1))
        route = convert_ordered_route(ordered_route,num_customers)
        # Check if the candidate column is optimal
        if reduced_cost < 0:
            print(ordered_route)
            print(reduced_cost)
            # Add the column to the master problem
            master_problem.add_columns([route], [cost], [ordered_route])
        else:
            # Optimality has been reached
            break

    # Extract the final solution from the master problem
    return master_problem.extract_solution()


def initialize_columns(num_customers, time_matrix):
    singular_routes = []
    costs = []
    ordered_routes = []
    for i in range(num_customers):
        route = np.zeros(num_customers)
        route[i] = 1
        singular_routes.append(route)
        ordered_routes.append((0, i, 0))
        costs.append(time_matrix[0][i + 1] + time_matrix[i + 1][0])

    return singular_routes, costs, ordered_routes


def convert_ordered_route(ordered_route, num_customers):
    route = np.zeros(num_customers)
    for customer in ordered_route:
        if customer != 0:
            route[customer - 1] = 1
    return route


class MasterProblem:
    def __init__(self, num_customers, initial_routes, costs, ordered_routes):

        self.num_customers = num_customers
        # Initialize the model
        self.build_model(initial_routes, costs, ordered_routes)

    def build_model(self, initial_routes, costs, ordered_routes):
        self.model = gb.Model()
        self.model.setParam('OutputFlag', False)
        self.y = {}
        self.routes = {}
        self.route_count = 0
        self.add_columns(initial_routes, costs, ordered_routes)

    def add_columns(self, routes, costs, ordered_routes):

        constrs = self.model.getConstrs()
        if len(constrs) == 0:
            for index, route in enumerate(routes):
                self.y[index] = self.model.addVar(ub=1, obj=costs[index])
                self.routes[index] = ordered_routes[index]
            self.model.addConstrs(gb.quicksum(routes[r][i] * self.y[r] for r in range(len(routes))) >= 1
                                  for i in range(self.num_customers))

        else:
            for index, route in enumerate(routes):
                self.y[self.route_count + index] = self.model.addVar(ub=1, obj=costs[index],
                                                                     column=gb.Column(route, constrs))
                self.routes[self.route_count + index] = ordered_routes[index]

        self.route_count += len(routes)
        self.model.update()
        self.model.reset()

    def solve(self):
        self.model.optimize()

    def retain_duals(self):
        return self.model.getAttr("Pi", self.model.getConstrs())

    def extract_solution(self):
        if self.model.getAttr("Status") == 2:
            pass


class Subproblem:
    def __init__(self, num_customers, vehicle_capacity, time_matrix, demands, time_windows, time_limit, duals):
        self.num_customers = num_customers
        self.vehicle_capacity = vehicle_capacity
        self.time_matrix = time_matrix
        self.demands = demands
        self.time_windows = time_windows
        self.time_limit = time_limit
        self.duals = duals
        self.price = np.zeros((num_customers + 1, num_customers + 1))

        for i in range(num_customers + 1):
            for j in range(num_customers + 1):
                if i != j:
                    if i != 0:
                        self.price[i, j] = time_matrix[i, j] - duals[i - 1]
                    else:
                        self.price[i, j] = time_matrix[i, j]

    def dynamic_program(self, start_point, current_label, unvisited_customers, remaining_time, remaining_capacity,
                        current_time, current_price):
        if remaining_time < 0 or remaining_capacity < 0:
            return [], math.inf
        if start_point != 0:
            if current_time < self.time_windows[start_point, 0] or current_time > self.time_windows[start_point, 1]:
                #print('busted')
                return [], math.inf

        if current_label[0] == 0 and current_label[-1] == 0 and len(current_label) > 1:
            return current_label, current_price

        best_label = []
        best_price = math.inf
        for j in unvisited_customers:
            if j != start_point:
                copy_label = current_label.copy()
                copy_unvisited = unvisited_customers.copy()
                RT = remaining_time
                RC = remaining_capacity
                CT = current_time
                CP = current_price

                copy_label.append(j)
                copy_unvisited.remove(j)
                RT -= self.time_matrix[start_point, j]
                RC -= self.demands[j]
                CT += self.time_matrix[start_point, j]
                CP += self.price[start_point, j]

                label, price = self.dynamic_program(j, copy_label, copy_unvisited, RT, RC, CT, CP)

                if price < best_price:
                    best_price = price
                    best_label = label

        return best_label, best_price

    def solve(self):
        start_point = 0
        current_label = [0]
        unvisited_customers = list(range(0, self.num_customers + 1))
        remaining_time = self.time_limit
        remaining_capacity = self.vehicle_capacity
        current_time = 0
        current_price = 0
        best_route, best_cost = self.dynamic_program(start_point, current_label, unvisited_customers, remaining_time,
                                                     remaining_capacity,
                                                     current_time, current_price)
        return best_route, best_cost


def main():
    num_customers = 10
    VRP_instance = Instance_Generator(num_customers)
    time_matrix = VRP_instance.time_matrix
    time_windows = VRP_instance.time_windows
    demands = VRP_instance.demands
    vehicle_capacity = VRP_instance.vehicle_capacity
    time_limit = VRP_instance.time_limit
    solve_vrp_with_time_windows(vehicle_capacity, time_matrix, demands, time_windows, time_limit, num_customers)


if __name__ == "__main__":
    main()
