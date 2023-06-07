import numpy as np
import gurobipy as gb
from instance_generator import Instance_Generator
import math
import random


def solve_relaxed_vrp_with_time_windows(vehicle_capacity, time_matrix, demands, time_windows, time_limit, num_customers,
                                        service_times, forbidden_edges, compelled_edges,
                                        initial_routes, initial_costs, initial_orders):
    # Ensure all input lists are of the same length
    assert len(time_matrix) == len(demands) == len(time_windows)

    if initial_routes == []:
        initial_routes, initial_costs, initial_orders = initialize_columns(num_customers, time_matrix)

    forbidden_edges = create_forbidden_edges_list(num_customers,forbidden_edges, compelled_edges)
    compelled_edges = []

    # Initialize the master problem
    master_problem = MasterProblem(num_customers, initial_routes, initial_costs, initial_orders, forbidden_edges,
                                   compelled_edges)

    added_orders = initial_orders.copy()
    # Iterate until optimality is reached
    while True:
        master_problem.solve()
        duals = master_problem.retain_duals()
        subproblem = Subproblem(num_customers, vehicle_capacity, time_matrix, demands, time_windows, time_limit,
                                duals, service_times, forbidden_edges)
        ordered_route, reduced_cost = subproblem.solve()
        print("RC is " + str(reduced_cost))
        print(ordered_route)
        cost = sum(time_matrix[ordered_route[i], ordered_route[i + 1]] for i in range(len(ordered_route) - 1))
        route = convert_ordered_route(ordered_route, num_customers)
        # Check if the candidate column is optimal
        if reduced_cost < 0 and ordered_route not in added_orders:
            # Add the column to the master problem
            master_problem.add_columns([route], [cost], [ordered_route], forbidden_edges, compelled_edges)
            added_orders.append(ordered_route)
        else:
            # Optimality has been reached
            print("Addition Failed")
            break

    sol, obj = master_problem.extract_solution()
    routes, costs, orders = master_problem.extract_columns()
    return sol, obj, routes, costs, orders


def initialize_columns(num_customers, time_matrix):
    singular_routes = []
    costs = []
    ordered_routes = []
    for i in range(num_customers):
        route = np.zeros(num_customers)
        route[i] = 1
        singular_routes.append(route)
        ordered_routes.append([0, i + 1, 0])
        costs.append(time_matrix[0][i + 1] + time_matrix[i + 1][0])

    return singular_routes, costs, ordered_routes


def convert_ordered_route(ordered_route, num_customers):
    route = np.zeros(num_customers)
    for customer in ordered_route:
        if customer != 0:
            route[customer - 1] = 1
    return route

def create_forbidden_edges_list(num_customers, forbidden_edges, compelled_edges):
    forbid_copy = forbidden_edges.copy()
    for edge in compelled_edges:
        forbid_copy += [[x, edge[1]] for x in range(num_customers + 1) if x != edge[0]]
    return forbid_copy

class MasterProblem:
    def __init__(self, num_customers, initial_routes, costs, ordered_routes, forbidden_edges, compelled_edges):

        self.num_customers = num_customers
        # Initialize the model
        self.build_model(initial_routes, costs, ordered_routes, forbidden_edges, compelled_edges)

    def build_model(self, initial_routes, costs, ordered_routes, forbidden_edges, compelled_edges):
        self.model = gb.Model()
        self.model.setParam('OutputFlag', False)
        self.y = {}
        self.routes = {}
        self.costs = {}
        self.orders = {}
        self.route_count = 0
        self.add_columns(initial_routes, costs, ordered_routes, forbidden_edges, compelled_edges)

    def add_columns(self, routes, costs, ordered_routes, forbidden_edges, compelled_edges):
        constrs = self.model.getConstrs()

        if len(constrs) == 0:
            to_be_imposed = {}
            for edge in compelled_edges:
                to_be_imposed[tuple(edge)] = []

            for index, route in enumerate(routes):
                var_ub = 1
                ordered_route = ordered_routes[index]
                self.routes[index] = route
                self.costs[index] = costs[index]
                self.orders[index] = ordered_route
                for index2 in range(len(ordered_route) - 1):
                    edge = [ordered_route[index2], ordered_route[index2 + 1]]
                    if edge in forbidden_edges:
                        var_ub = 0
                    elif edge in compelled_edges:
                        to_be_imposed[tuple(edge)].append(index)
                self.y[index] = self.model.addVar(ub=var_ub, obj=costs[index])

            self.model.addConstrs(gb.quicksum(routes[r][i] * self.y[r] for r in range(len(routes))) >= 1
                                  for i in range(self.num_customers))
            self.model.addConstrs(gb.quicksum(self.y[t] for t in to_be_imposed[tuple(compelled_edges[k])]) >= 1
                                  for k in range(len(compelled_edges)))
        else:
            for index, route in enumerate(routes):
                var_ub = 1
                column_contin = np.array([0] * len(compelled_edges))
                ordered_route = ordered_routes[index]
                self.routes[self.route_count + index] = route
                self.costs[self.route_count + index] = costs[index]
                self.orders[self.route_count + index] = ordered_route
                for index2 in range(len(ordered_route) - 1):
                    edge = [ordered_route[index2], ordered_route[index2 + 1]]
                    if edge in forbidden_edges:
                        var_ub = 0
                    elif edge in compelled_edges:
                        index3 = compelled_edges.index(edge)
                        column_contin[index3] = 1
                self.y[self.route_count + index] = self.model.addVar(ub=var_ub, obj=costs[index],
                                                                     column=gb.Column(
                                                                         np.concatenate((route, column_contin)),
                                                                         constrs))

        self.route_count += len(routes)
        self.model.update()
        self.model.reset()

    def solve(self):
        self.model.optimize()

    def retain_duals(self):
        return self.model.getAttr("Pi", self.model.getConstrs())

    def extract_solution(self):
        try:
            return [(key, self.y[key].x, self.orders[key]) for key in self.y if self.y[key].x > 0], self.model.objval
        except:
            print(self.model.getAttr("Status"))
            return [],math.inf


    def extract_columns(self):
        routes = [self.routes[x] for x in self.routes]
        costs = [self.costs[x] for x in self.costs]
        orders = [self.orders[x] for x in self.orders]
        return routes, costs, orders


class Subproblem:
    def __init__(self, num_customers, vehicle_capacity, time_matrix, demands, time_windows, time_limit, duals,
                 service_times, forbidden_edges):
        self.num_customers = num_customers
        self.vehicle_capacity = vehicle_capacity
        self.time_matrix = time_matrix
        self.demands = demands
        self.time_windows = time_windows
        self.time_limit = time_limit
        self.duals = duals
        self.service_times = service_times
        self.price = np.zeros((num_customers + 1, num_customers + 1))
        self.forbidden_edges = forbidden_edges

        for i in range(num_customers + 1):
            for j in range(num_customers + 1):
                if i != j:
                    if i != 0:
                        self.price[i, j] = time_matrix[i, j] - duals[i - 1]
                    else:
                        self.price[i, j] = time_matrix[i, j]

    def dynamic_program(self, start_point, current_label, unvisited_customers, remaining_time, remaining_capacity,
                        current_time, current_price):

        if current_time > self.time_windows[start_point, 1]:
            return [], math.inf
        if start_point != 0:
            if current_time < self.time_windows[start_point, 0]:
                current_time = self.time_windows[start_point, 0]

        current_time += self.service_times[start_point]
        remaining_time -= self.service_times[start_point]

        if remaining_time < 0 or remaining_capacity < 0:
            return [], math.inf

        if current_label[0] == 0 and current_label[-1] == 0 and len(current_label) > 1:
            return current_label, current_price

        best_label = []
        best_price = math.inf
        for j in unvisited_customers:
            if j != start_point and [start_point,j] not in self.forbidden_edges:
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
                                                     remaining_capacity, current_time, current_price)
        return best_route, best_cost


def main():
    random.seed(5)
    np.random.seed(25)
    num_customers = 12
    VRP_instance = Instance_Generator(num_customers)
    time_matrix = VRP_instance.time_matrix
    time_windows = VRP_instance.time_windows
    demands = VRP_instance.demands
    vehicle_capacity = VRP_instance.vehicle_capacity
    time_limit = VRP_instance.time_limit
    service_times = VRP_instance.service_times
    forbidden_edges = []
    compelled_edges = []
    initial_routes = []
    initial_costs = []
    initial_orders = []
    sol, obj, routes, costs, orders = solve_relaxed_vrp_with_time_windows(vehicle_capacity, time_matrix, demands,
                                                                          time_windows, time_limit,
                                                                          num_customers, service_times, forbidden_edges,
                                                                          compelled_edges,
                                                                          initial_routes, initial_costs, initial_orders)
    print(sol)
    print(obj)


if __name__ == "__main__":
    main()
