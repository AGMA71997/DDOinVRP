import sys

import numpy as np
import gurobipy as gb
from instance_generator import Instance_Generator
import math
import random
import time
from threading import Thread


def solve_relaxed_vrp_with_time_windows(vehicle_capacity, time_matrix, demands, time_windows, time_limit, num_customers,
                                        service_times, forbidden_edges, compelled_edges,
                                        initial_routes, initial_costs, initial_orders):
    # Ensure all input lists are of the same length
    assert len(time_matrix) == len(demands) == len(time_windows)

    if initial_routes == []:
        initial_routes, initial_costs, initial_orders = initialize_columns(num_customers, vehicle_capacity, time_matrix,
                                                                           service_times, time_windows,
                                                                           demands)
        total_cost = sum(initial_costs[x] for x in range(len(initial_costs)))
        print("The initial routes: " + str(initial_orders))
        print("with total cost: " + str(total_cost))
        for route in initial_orders:
            if not check_route_feasibility(route, time_matrix, time_windows, service_times, demands, vehicle_capacity):
                print("Infeasible route detected")
                sys.exit(0)

    forbidden_edges = create_forbidden_edges_list(num_customers, forbidden_edges, compelled_edges)
    compelled_edges = []

    # Initialize the master problem
    master_problem = MasterProblem(num_customers, initial_routes, initial_costs, initial_orders, forbidden_edges,
                                   compelled_edges)

    added_orders = initial_orders

    # Iterate until optimality is reached
    while True:
        master_problem.solve()
        duals = master_problem.retain_duals()
        time_11 = time.time()
        subproblem = Subproblem(num_customers, vehicle_capacity, time_matrix, demands, time_windows, time_limit,
                                duals, service_times, forbidden_edges)
        ordered_route, reduced_cost = subproblem.solve()
        time_22 = time.time()
        print("RC is " + str(reduced_cost))
        print("Total solving time for PP is: " + str(time_22 - time_11))
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


def initialize_columns(num_customers, truck_capacity, time_matrix, service_times, time_windows, demands):
    unvisited_customers = list(range(1, num_customers + 1))
    solution = []
    current_stop = 0
    current_route = [0]
    remaining_capacity = truck_capacity
    current_time = 0
    while len(unvisited_customers) > 0:
        nearest_customers = np.argsort(time_matrix[current_stop, :].copy())
        i = 0
        feasible_addition = False

        while not feasible_addition:
            new_stop = nearest_customers[i]
            waiting_time = max(time_windows[new_stop, 0] - (current_time + time_matrix[current_stop, new_stop]), 0)
            total_return_time = time_matrix[current_stop, new_stop] + waiting_time + service_times[new_stop] + \
                                time_matrix[new_stop, 0]
            if current_time + time_matrix[current_stop, new_stop] > \
                    time_windows[new_stop, 1] or remaining_capacity < demands[
                new_stop] or new_stop not in unvisited_customers or current_time + total_return_time > \
                    time_windows[0, 1]:
                i += 1
            else:
                current_route.append(new_stop)
                remaining_capacity -= demands[new_stop]
                current_time = max(current_time + time_matrix[current_stop, new_stop],
                                   time_windows[new_stop, 0]) + service_times[new_stop]
                unvisited_customers.remove(new_stop)
                current_stop = new_stop
                feasible_addition = True

            if not feasible_addition and i == num_customers + 1:
                current_route.append(0)
                solution.append(current_route)
                current_stop = 0
                current_route = [0]
                remaining_capacity = truck_capacity
                current_time = 0
                break

    current_route.append(0)
    solution.append(current_route)

    singular_routes = []
    costs = []
    for route in solution:
        singular_routes.append(convert_ordered_route(route, num_customers))
        costs.append(sum(time_matrix[route[i], route[i + 1]] for i in range(len(route) - 1)))
    return singular_routes, costs, solution


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


def check_route_feasibility(route, dist_matrix_data, time_windows, service_times, demands_data, truck_capacity):
    current_time = max(dist_matrix_data[0, route[1]], time_windows[route[1], 0])
    total_capacity = 0

    for i in range(1, len(route)):
        if current_time > time_windows[route[i], 1]:
            #print("Time Window violated")
            #print(route[i])
            return False
        current_time += service_times[route[i]]
        total_capacity += demands_data[route[i]]
        if total_capacity > truck_capacity:
            #print("Truck Capacity Violated")
            return False
        if i < len(route) - 1:
            # travel to next node
            current_time += dist_matrix_data[route[i], route[i + 1]]
            current_time = max(current_time, time_windows[route[i + 1], 0])
    return True


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
            return [], math.inf

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
        self.service_times = service_times
        self.forbidden_edges = forbidden_edges

        duals.insert(0, 0)
        duals = np.array(duals)
        duals = duals.reshape((len(duals), 1))
        self.price = time_matrix - duals

        self.determine_PULSE_bounds(2)

        # route = [0, 6, 3, 1, 0]
        # print(sum(self.price[route[x], route[x + 1]] for x in range(len(route) - 1)))
        # print(check_route_feasibility(route, time_matrix, time_windows, service_times, demands, vehicle_capacity))

    def determine_PULSE_bounds(self, increment):
        self.increment = increment
        self.no_of_increments = math.ceil(self.time_windows[0, 1] / self.increment - 1)
        self.bounds = np.zeros((self.num_customers, self.no_of_increments)) + math.inf
        self.supreme_labels = {}
        self.supreme_capacities = {}

        for inc in range(self.no_of_increments, 4, -1):
            threads = []
            #print(inc)
            for cus in range(1, self.num_customers + 1):
                start_point = cus
                current_label = [cus]
                unvisited_customers = list(range(0, self.num_customers + 1))
                unvisited_customers.remove(cus)
                remaining_capacity = self.vehicle_capacity - self.demands[cus]
                current_time = self.time_windows[0, 1] - (self.no_of_increments - inc + 1) * increment
                current_price = 0
                best_bound = math.inf
                thread = Bound_Threader(target=self.bound_calculator, args=(start_point, current_label,
                                                                            unvisited_customers, remaining_capacity,
                                                                            current_time, current_price,
                                                                            best_bound))
                thread.start()
                threads.append(thread)

            for index, thread in enumerate(threads):
                label, lower_bound = thread.join()
                self.bounds[index, inc - 1] = lower_bound
                self.supreme_labels[index + 1, inc] = label

    def bound_calculator(self, start_point, current_label, unvisited_customers,
                         remaining_capacity, current_time, current_price, best_bound):

        if current_time > self.time_windows[start_point, 1] or remaining_capacity < 0:
            return [], math.inf

        if start_point == 0 and len(current_label) > 1:
            return current_label, current_price

        waiting_time = max(self.time_windows[start_point, 0] - current_time, 0)
        current_time += waiting_time
        current_time += self.service_times[start_point]

        inc = math.ceil(self.no_of_increments - (self.time_windows[0, 1] - current_time) / self.increment)
        if 0 < inc <= self.no_of_increments:
            if self.bounds[start_point - 1, inc - 1] < math.inf:
                bound_estimate = current_price + self.bounds[start_point - 1, inc - 1]
                pro_route = current_label[:-1] + self.supreme_labels[start_point, inc]
                if bound_estimate > best_bound:
                    return [], math.inf

        best_label = []
        for j in unvisited_customers:
            if j != start_point and [start_point, j] not in self.forbidden_edges:

                copy_label = current_label.copy()
                copy_unvisited = unvisited_customers.copy()
                RC = remaining_capacity
                CT = current_time
                CP = current_price

                copy_label.append(j)
                copy_unvisited.remove(j)
                RC -= self.demands[j]
                CT += self.time_matrix[start_point, j]
                CP += self.price[start_point, j]

                if len(copy_label) > 2 and j != 0:
                    roll_back_price = CP - (self.price[copy_label[-3], start_point] + self.price[start_point, j]) + \
                                      self.price[copy_label[-3], j]

                    roll_back_time = CT - (
                            self.time_matrix[start_point, j] + self.service_times[start_point] + waiting_time +
                            self.time_matrix[copy_label[-3], start_point])
                    roll_back_time += self.time_matrix[copy_label[-3], j]
                    roll_back_time = max(roll_back_time, self.time_windows[j, 0])

                    if roll_back_price <= CP and roll_back_time <= max(self.time_windows[j, 0], CT):
                        CT = math.inf

                label, lower_bound = self.bound_calculator(j, copy_label, copy_unvisited, RC, CT, CP,
                                                           best_bound)
                if lower_bound < best_bound:
                    best_bound = lower_bound
                    best_label = label

        return best_label, best_bound

    def dynamic_program(self, start_point, current_label, unvisited_customers, remaining_capacity,
                        current_time, current_price):

        if current_time > self.time_windows[start_point, 1] or remaining_capacity < 0:
            return [], math.inf

        if current_label[-1] == 0 and len(current_label) > 1:
            return current_label, current_price

        current_time = max(self.time_windows[start_point, 0], current_time)
        current_time += self.service_times[start_point]

        best_label = []
        best_price = math.inf
        for j in unvisited_customers:
            if j != start_point and [start_point, j] not in self.forbidden_edges:
                copy_label = current_label.copy()
                copy_unvisited = unvisited_customers.copy()
                RC = remaining_capacity
                CT = current_time
                CP = current_price

                copy_label.append(j)
                copy_unvisited.remove(j)
                RC -= self.demands[j]
                CT += self.time_matrix[start_point, j]
                CP += self.price[start_point, j]

                label, price = self.dynamic_program(j, copy_label, copy_unvisited, RC, CT, CP)

                if price < best_price:
                    best_price = price
                    best_label = label

        return best_label, best_price

    def solve(self):
        start_point = 0
        current_label = [0]
        unvisited_customers = list(range(0, self.num_customers + 1))
        remaining_capacity = self.vehicle_capacity
        current_time = 0
        current_price = 0
        best_bound = math.inf
        best_route, best_cost = self.bound_calculator(start_point, current_label, unvisited_customers,
                                                      remaining_capacity, current_time, current_price,
                                                      best_bound)
        return best_route, best_cost


class Bound_Threader(Thread):

    def __init__(self, target, args):
        Thread.__init__(self, target=target, args=args)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self):
        Thread.join(self)
        return self._return


def main():
    random.seed(5)
    np.random.seed(25)
    num_customers = 15
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
    time_1 = time.time()

    sol, obj, routes, costs, orders = solve_relaxed_vrp_with_time_windows(vehicle_capacity, time_matrix, demands,
                                                                          time_windows, time_limit,
                                                                          num_customers, service_times, forbidden_edges,
                                                                          compelled_edges,
                                                                          initial_routes, initial_costs, initial_orders)
    time_2 = time.time()

    print("time: " + str(time_2 - time_1))
    print("solution: " + str(sol))
    print("objective: " + str(obj))
    print("number of columns: " + str(len(orders)))


if __name__ == "__main__":
    main()
