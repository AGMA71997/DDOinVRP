import sys

import numpy as np
import gurobipy as gb
import torch

from instance_generator import Instance_Generator
import random
import time
from threading import Thread

from utils import *
import argparse
import json

import matplotlib.pyplot as pp
from graph_reduction import Node_Reduction, Arc_Reduction


def solve_relaxed_vrp_with_time_windows(VRP_instance, forbidden_edges, compelled_edges, initial_routes,
                                        initial_costs, initial_orders, policy, arc_red):
    coords = VRP_instance.coords
    time_matrix = VRP_instance.time_matrix
    time_windows = VRP_instance.time_windows
    demands = VRP_instance.demands
    vehicle_capacity = VRP_instance.vehicle_capacity
    service_times = VRP_instance.service_times
    num_customers = VRP_instance.N

    # Ensure all input lists are of the same length
    assert len(time_matrix) == len(demands) == len(time_windows)

    if not initial_routes:
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
    reoptimize = True
    max_iter = 5000
    max_time = 60 * 60
    start_time = time.time()
    results_dict = {}
    iteration = 0
    cum_time = 0
    arc_red = arc_red
    while iteration < max_iter:

        if time.time() - start_time > max_time:
            print("Time Limit Reached")
            break

        master_problem.solve()
        duals = master_problem.retain_duals()

        prices = create_price(time_matrix, duals) * -1

        NR = Node_Reduction(coords, duals)
        red_cor = NR.dual_based_elimination()
        red_cor, red_dem, red_tws, red_duals, red_sts, red_tms, red_prices, cus_mapping = reshape_problem(red_cor,
                                                                                                          demands,
                                                                                                          time_windows,
                                                                                                          duals,
                                                                                                          service_times,
                                                                                                          time_matrix,
                                                                                                          prices)

        N = len(red_cor) - 1
        time_11 = time.time()
        if heuristic:
            subproblem = Subproblem(N, vehicle_capacity, red_tms, red_dem, red_tws,
                                    red_duals, red_sts, forbidden_edges, red_prices)
            ordered_route, reduced_cost, top_labels = subproblem.solve_heuristic(arc_red=arc_red, policy=policy,
                                                                                 max_threads=100)
        else:
            subproblem = Subproblem(N, vehicle_capacity, red_tms, red_dem, red_tws,
                                    red_duals, red_sts, forbidden_edges, red_prices)
            ordered_route, reduced_cost, top_labels = subproblem.solve()
        time_22 = time.time()
        # subproblem.render_solution(ordered_route)

        ordered_route = remap_route(ordered_route, cus_mapping)

        cost = sum(time_matrix[ordered_route[i], ordered_route[i + 1]] for i in range(len(ordered_route) - 1))
        route = convert_ordered_route(ordered_route, num_customers)

        iteration += 1
        obj_val = master_problem.model.objval
        cum_time += time_22 - time_11
        if iteration % 10 == 0:
            print("Iteration: " + str(iteration))
            # print("Solving time for PP is: " + str(time_22 - time_11))
            print("RC is " + str(reduced_cost))
            print("Best route: " + str(ordered_route))
            print("The objective value is: " + str(obj_val))
            print("The total number of generated columns is: " + str(len(top_labels) + 1))
            print("The total time spent on PP is: " + str(cum_time))
            results_dict[iteration] = (obj_val, time.time() - start_time)

        # Check if the candidate column is optimal
        if reduced_cost < -0.001:
            # Add the column to the master problem
            master_problem.add_columns([route], [cost], [ordered_route], forbidden_edges, compelled_edges)
            added_orders.append(ordered_route)
            for x in range(len(top_labels)):
                label = top_labels[x]
                label = remap_route(label, cus_mapping)
                cost = sum(time_matrix[label[i], label[i + 1]] for i in range(len(label) - 1))
                route = convert_ordered_route(label, num_customers)
                master_problem.add_columns([route], [cost], [label], forbidden_edges, compelled_edges)
                added_orders.append(label)
        elif arc_red:
            print("Arc red mode to be changed.")
            break
            # arc_red = False
        else:
            # Optimality has been reached
            reoptimize = False
            print("Addition Failed")
            break

    if reoptimize:
        master_problem.solve()
    sol, obj = master_problem.extract_solution()
    results_dict["Final"] = (obj, time.time() - start_time)
    routes, costs, orders = master_problem.extract_columns()
    master_problem.__delete__()
    return sol, obj, routes, costs, orders, results_dict


class MasterProblem:
    def __init__(self, num_customers, initial_routes, costs, ordered_routes, forbidden_edges, compelled_edges):

        self.num_customers = num_customers
        # Initialize the model
        self.build_model(initial_routes, costs, ordered_routes, forbidden_edges, compelled_edges)

    def build_model(self, initial_routes, costs, ordered_routes, forbidden_edges, compelled_edges):
        self.env = gb.Env()
        self.model = gb.Model(env=self.env)
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
            print("The model failed with status:" + str(self.model.getAttr("Status")))
            return [], math.inf

    def extract_columns(self):
        routes = [self.routes[x] for x in self.routes]
        costs = [self.costs[x] for x in self.costs]
        orders = [self.orders[x] for x in self.orders]
        return routes, costs, orders

    def __delete__(self):
        self.env.dispose()
        self.model.dispose()


class Subproblem:
    def __init__(self, num_customers, vehicle_capacity, time_matrix, demands, time_windows, duals,
                 service_times, forbidden_edges, prices=None):
        self.num_customers = num_customers
        self.vehicle_capacity = vehicle_capacity
        self.time_matrix = time_matrix
        self.demands = demands
        self.time_windows = time_windows

        self.service_times = service_times
        self.forbidden_edges = forbidden_edges
        self.duals = duals

        if prices is None:
            self.price = create_price(time_matrix, duals) * -1
        else:
            self.price = prices

        self.price_arrangement = self.arrange_per_price()

    def arrange_per_price(self):
        arrangements = {}
        for cus in range(self.num_customers + 1):
            arrangements[cus] = numpy.argsort(self.price[cus, :])
        return arrangements

    def determine_PULSE_bounds(self, increment, stopping_time):
        self.increment = increment
        self.no_of_increments = math.ceil(self.time_windows[0, 1] / self.increment - 1)
        self.bounds = np.zeros((self.num_customers, self.no_of_increments)) + math.inf
        self.supreme_labels = {}
        stopping_inc = math.ceil(stopping_time / self.increment - 1)

        for inc in range(self.no_of_increments, stopping_inc, -1):
            threads = []
            for cus in self.price_arrangement[0]:
                if cus == 0:
                    continue
                start_point = cus
                current_label = [cus]
                remaining_capacity = self.vehicle_capacity - self.demands[cus]
                current_time = self.time_windows[0, 1] - (self.no_of_increments - inc + 1) * increment
                current_price = 0
                best_bound = min(numpy.min(self.bounds[cus - 1, :]), 0)
                solve = False
                thread = Bound_Threader(target=self.bound_calculator, args=(start_point, current_label,
                                                                            remaining_capacity,
                                                                            current_time, current_price,
                                                                            best_bound, solve))
                thread.start()
                threads.append(thread)

            for index, thread in enumerate(threads):
                label, lower_bound = thread.join()
                self.bounds[index, inc - 1] = lower_bound
                self.supreme_labels[index + 1, inc] = label

    def bound_calculator(self, start_point, current_label, remaining_capacity, current_time,
                         current_price, best_bound, solve):

        if current_time > self.time_windows[start_point, 1] or remaining_capacity < 0:
            return [], math.inf

        if start_point == 0 and len(current_label) > 1:
            if current_price < -0.001:
                if solve:
                    if current_price < self.primal_bound:
                        self.primal_bound = current_price
                        self.primal_label = current_label
            else:
                current_price = 0
            return current_label, current_price

        waiting_time = max(self.time_windows[start_point, 0] - current_time, 0)
        current_time += waiting_time
        current_time += self.service_times[start_point]

        inc = math.ceil(self.no_of_increments - (self.time_windows[0, 1] - current_time) / self.increment)
        if 0 < inc <= self.no_of_increments:
            if self.bounds[start_point - 1, inc - 1] < math.inf:
                bound_estimate = current_price + self.bounds[start_point - 1, inc - 1]
                if solve:
                    if bound_estimate > self.primal_bound:
                        return [], math.inf
                else:
                    if bound_estimate > best_bound:
                        return [], math.inf

        best_label = []
        best_price_indices = self.price_arrangement[start_point]
        for index in range(len(best_price_indices)):
            j = best_price_indices[index]
            if j > 0:
                if j in current_label:
                    continue
            else:
                if start_point == 0:
                    continue

            if [start_point, j] not in self.forbidden_edges and self.price[start_point, j] != math.inf:

                copy_label = current_label.copy()
                RC = remaining_capacity
                CT = current_time
                CP = current_price

                copy_label.append(j)
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

                label, lower_bound = self.bound_calculator(j, copy_label, RC, CT, CP,
                                                           best_bound, solve)

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
        best_price_indices = self.price_arrangement[start_point]
        for index in range(len(best_price_indices)):
            j = best_price_indices[index]
            if j in unvisited_customers:
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

    def DP_heuristic(self, start_point, current_label, remaining_capacity, current_time,
                     current_price, best_bound, start_time, thread_time_limit=3, price_lb=-0.1):
        if start_time is None:
            start_time = time.time()

        terminate = self.terminate
        if time.time() - start_time > thread_time_limit:
            terminate = True
            self.thread_count += 1
            # print("Thread " + str(current_label[1]) + " failed.")
            if self.thread_count == self.max_threads:
                self.terminate = True

        if current_time > self.time_windows[start_point, 1] or remaining_capacity < 0 or terminate \
                or (current_price > 0 and current_time / self.time_windows[0, 1] > 0.75) or \
                (current_price > 0 and remaining_capacity / self.vehicle_capacity < 0.25):
            return [], math.inf, terminate

        if start_point == 0 and len(current_label) > 1:
            if current_price < price_lb:
                self.col_count += 1
                self.thread_count += 1
                terminate = True
                # print("Thread " + str(current_label[1]) + " succeeded.")
                if self.col_count == self.max_columns or self.thread_count == self.max_threads:
                    self.terminate = True
            return current_label, current_price, terminate

        waiting_time = max(self.time_windows[start_point, 0] - current_time, 0)
        current_time += waiting_time
        current_time += self.service_times[start_point]

        best_label = []
        for j in self.price_arrangement[start_point]:
            if j > 0:
                if j in current_label:
                    continue
            else:
                if start_point == 0:
                    continue

            if [start_point, j] not in self.forbidden_edges and self.price[start_point, j] != math.inf:

                copy_label = current_label.copy()
                RC = remaining_capacity
                CT = current_time
                CP = current_price

                copy_label.append(j)
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

                label, lower_bound, terminate = self.DP_heuristic(j, copy_label, RC, CT, CP,
                                                                  best_bound, start_time, thread_time_limit,
                                                                  price_lb)

                if lower_bound < best_bound:
                    best_bound = lower_bound
                    best_label = label

                if terminate:
                    break

        return best_label, best_bound, terminate

    def k_exchange(self, path, dist):
        path_length = len(path)
        new_path_list = []
        try:
            index = int(torch.randint(1, path_length - 1, ()))
        except:
            print("Infeasible Route Detected")
            print(path)
            sys.exit(0)

        chosen = path[index]
        dist_sum = sum([torch.sum(dist[node, :]) for node in path if node != 0])
        if dist_sum == 0:
            return [[]]
        while torch.sum(dist[chosen]) == 0:
            index = int(torch.randint(1, path_length - 1, ()))
            chosen = path[index]

        randi = torch.rand(())
        connect = None
        for j in range(len(dist[chosen])):
            if randi < dist[chosen, j] and (chosen, j) not in self.forbidden_edges \
                    and self.price[chosen, j] != math.inf:
                connect = j
                break

        if connect is None:
            return [[]]
        elif connect not in path:
            new_path = path[:index + 1] + [connect] + path[index + 1:]
            new_path_list.append(new_path)
        else:
            if connect == 0:
                new_path = path[:index + 1] + [0]
                new_path_list.append(new_path)
            else:
                connect_index = path.index(connect)
                new_path = path[:]
                if path[index + 1] != 0:
                    new_path[index + 1], new_path[connect_index] = new_path[connect_index], new_path[index + 1]
                    new_path_list.append(new_path)

                if connect_index == index + 2:
                    new_path = path[:index + 1] + path[index + 2:]
                    new_path_list.append(new_path)

        return new_path_list

    def h_ls(self, dist, k_opt_iter, start_point, current_label, remaining_capacity, current_time,
             current_price, best_bound, start_time, TTL, PLB):
        """Local search algorithm H_ls for RCESPP."""
        current_path, current_cost, terminate = self.DP_heuristic(start_point, current_label,
                                                                  remaining_capacity, current_time,
                                                                  current_price, best_bound,
                                                                  start_time, TTL, PLB)
        init_cost = current_cost

        if not current_path or self.k_ex_count >= self.max_columns:
            return [], 0

        for k in range(k_opt_iter):
            neighbors = self.k_exchange(current_path, dist)

            for neighbor in neighbors:
                if check_route_feasibility(neighbor, self.time_matrix, self.time_windows,
                                           self.service_times, self.demands, self.vehicle_capacity):
                    neighbor_cost = sum(
                        [self.price[neighbor[x], neighbor[x + 1]] for x in range(len(neighbor) - 1)])
                    if neighbor_cost < current_cost:
                        current_path = neighbor
                        current_cost = neighbor_cost

        if current_cost < PLB:
            self.k_ex_count += 1
            if PLB < init_cost:
                self.col_count += 1
                if self.col_count == self.max_columns:
                    self.terminate = True
        return current_path, current_cost

    def solve_heuristic(self, arc_red=False, policy="DP", max_columns=20, max_threads=None,
                        k_opt_iter=100, dist=None):
        if policy == "DP" or "k-opt":
            self.terminate = False
            self.col_count = 0
            self.max_columns = min(self.num_customers, max_columns)
            self.thread_count = 0
            if max_threads is None:
                self.max_threads = self.num_customers
            else:
                self.max_threads = max_threads

            if policy == "DP" and arc_red:
                AR = Arc_Reduction(self.price, self.duals)
                self.price = AR.BE2()

            threads = []
            best_routes = []
            best_costs = []
            for cus in self.price_arrangement[0]:
                start_point = cus
                if cus != 0 and (0, cus) not in self.forbidden_edges and self.price[0, cus] != math.inf:
                    current_label = [0, cus]
                    remaining_capacity = self.vehicle_capacity - self.demands[cus]
                    current_time = self.time_matrix[0, cus]
                    current_price = self.price[0, cus]
                    start_time = None
                    if policy == "DP":
                        best_bound = 0 # math.inf
                        TTL = 30
                        PLB = -1
                        thread = Bound_Threader(target=self.DP_heuristic, args=(start_point, current_label,
                                                                                remaining_capacity,
                                                                                current_time, current_price,
                                                                                best_bound, start_time, TTL,
                                                                                PLB))
                    else:
                        dist = dist + torch.transpose(dist, 0, 1)
                        row_sums = dist.sum(axis=1, keepdims=True)
                        row_sums[row_sums == 0] = 1
                        dist = dist / row_sums
                        dist = torch.cumsum(dist, dim=1)

                        self.k_ex_count = 0
                        best_bound = 0.5
                        TTL = 5
                        PLB = -1
                        thread = Bound_Threader(target=self.h_ls, args=(dist, k_opt_iter, start_point,
                                                                        current_label, remaining_capacity,
                                                                        current_time, current_price,
                                                                        best_bound, start_time, TTL,
                                                                        PLB))

                    thread.start()
                    threads.append(thread)

            for index, thread in enumerate(threads):
                if policy == "DP":
                    label, cost, terminate = thread.join()
                else:
                    label, cost = thread.join()
                best_routes.append(label)
                best_costs.append(cost)

            if len(best_costs) > 0:
                price = min(best_costs)
                best_index = best_costs.index(price)
                label = best_routes[best_index]
                best_routes.remove(label)
                best_costs.remove(price)
                promising_labels = [best_routes[x] for x in range(len(best_routes)) if best_costs[x] < -0.001]
            else:
                label, promising_labels = [], []
                price = 0
        else:
            print("Heuristic not Implemented")
            sys.exit(0)

        return label, price, promising_labels

    def solve(self):
        self.primal_bound = 0
        self.primal_label = []

        self.determine_PULSE_bounds(0.5, 0.75 * self.time_windows[0, 1])
        print("Bounds Computed")

        threads = []
        best_routes = []
        best_costs = []
        self.primal_bound = min(numpy.min(self.bounds), 0)
        for cus in self.price_arrangement[0]:
            start_point = cus
            if cus != 0 and (0, cus) not in self.forbidden_edges and self.price[0, cus] != math.inf:
                current_label = [0, cus]
                remaining_capacity = self.vehicle_capacity - self.demands[cus]
                current_time = self.time_matrix[0, cus]
                current_price = self.price[0, cus]
                best_bound = 0
                solve = True
                thread = Bound_Threader(target=self.bound_calculator, args=(start_point, current_label,
                                                                            remaining_capacity,
                                                                            current_time, current_price,
                                                                            best_bound, solve))
                thread.start()
                threads.append(thread)

        for index, thread in enumerate(threads):
            label, cost = thread.join()
            best_routes.append(label)
            best_costs.append(cost)

        best_cost = min(best_costs)
        best_index = best_costs.index(best_cost)
        best_route = best_routes[best_index]
        best_routes.remove(best_route)
        best_costs.remove(best_cost)
        promising_labels = [best_routes[x] for x in range(len(best_routes)) if best_costs[x] < -0.001]
        return best_route, best_cost, promising_labels

    def render_solution(self, solution):
        pass


class Bound_Threader(Thread):

    def __init__(self, target, args):
        Thread.__init__(self, target=target, args=args)
        self._return = None
        self.daemon = True

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self):
        Thread.join(self)
        return self._return


def main():
    random.seed(10)
    np.random.seed(10)
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_customers', type=int, default=200)
    parser.add_argument('--policy', type=str, default='DP')
    parser.add_argument('--AR', type=bool, default=True)
    args = parser.parse_args()
    num_customers = args.num_customers
    policy = args.policy
    arc_red = args.AR

    global heuristic
    heuristic = True

    file = "config.json"
    with open(file, 'r') as f:
        config = json.load(f)

    results = []
    performance_dicts = []
    # directory = config["Solomon Test Dataset"]
    # for instance in os.listdir(directory):
    for experiment in range(10):
        # file = directory + "/" + instance
        # file = directory + "/" + "C206.txt"
        # print(file)

        VRP_instance = Instance_Generator(N=num_customers)
        # VRP_instance = Instance_Generator(file_path=file, config=config)

        print("This instance has " + str(num_customers) + " customers.")
        forbidden_edges = []
        compelled_edges = []
        initial_routes = []
        initial_costs = []
        initial_orders = []

        sol, obj, routes, costs, orders, results_dict = solve_relaxed_vrp_with_time_windows(VRP_instance,
                                                                                            forbidden_edges,
                                                                                            compelled_edges,
                                                                                            initial_routes,
                                                                                            initial_costs,
                                                                                            initial_orders,
                                                                                            policy, arc_red)

        print("solution: " + str(sol))
        print("objective: " + str(obj))
        print("number of columns: " + str(len(orders)))

        results.append(obj)
        performance_dicts.append(results_dict)

    mean_obj = statistics.mean(results)
    std_obj = statistics.stdev(results)
    print("The mean objective value is: " + str(mean_obj))
    print("The std dev. objective is: " + str(std_obj))

    pickle_out = open('DP Results N=' + str(num_customers) + ' ' + str(arc_red) + ' Large Scale Instances', 'wb')
    pickle.dump(performance_dicts, pickle_out)
    pickle_out.close()


if __name__ == "__main__":
    import cProfile

    # cProfile.run('main()')
    main()
