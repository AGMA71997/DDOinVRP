import pickle
import os
import numpy
import torch
import math
import matplotlib.pyplot as pp
import scipy.stats as st
import sys
import statistics


def result_analyzer(method, base, num_customers, addition=None, switch=False):
    if not os.getcwd().endswith('results'):
        os.chdir('results')

    file_name = method + ' Results N=' + str(num_customers) + ' ' + addition
    pickle_in = open(file_name, 'rb')
    performance_dicts = pickle.load(pickle_in)

    pickle_in = open('DP Results N=' + str(num_customers) + ' ' + base, 'rb')
    baseline = pickle.load(pickle_in)
    unmatched = 0

    if switch:
        per_dict_copy = performance_dicts.copy()
        performance_dicts = baseline
        baseline = per_dict_copy

    time_obj_map = {}
    if method == "RL":
        Time_limit = 700
        for x in range(0, Time_limit, 1):
            time_obj_map[x] = []
    else:
        Time_limit = 3600
        for x in range(0, Time_limit, 50):
            time_obj_map[x] = []

    DP_catch_up = []
    Gaps = []
    method_run_times = []
    method_CG_iters = []
    baseline_CG_iters = []
    method_PP_time = []
    baseline_PP_time = []
    for index, instance_dict in enumerate(performance_dicts):
        baseline_dict = baseline[index]
        skip = False
        # print(instance_dict)
        # print(baseline_dict)
        # print("---------------------------")

        Gaps.append((instance_dict["Final"][0] - baseline_dict["Final"][0]) * 100 / baseline_dict["Final"][0])
        if (instance_dict["Final"][0] - baseline_dict["Final"][0]) * 100 / baseline_dict["Final"][0] > 20:
            skip = True
        method_run_times.append(instance_dict["Final"][1])
        method_CG_iters.append(len(instance_dict) * 10)
        baseline_CG_iters.append(len(baseline_dict) * 10)
        method_PP_time.append(instance_dict["Final"][1] * 0.9 / (len(instance_dict) * 10))
        baseline_PP_time.append((baseline_dict["Final"][1] * 0.9 / (len(baseline_dict) * 10)))
        for key in instance_dict:
            time = instance_dict[key][1]
            obj_gap = (instance_dict[key][0] - baseline_dict["Final"][0]) * 100 / baseline_dict["Final"][0]
            for key2 in time_obj_map:
                if time < key2:
                    time_obj_map[key2].append(obj_gap)
                    break

        algo_final = instance_dict['Final'][0]
        stoppage = None
        for key in baseline_dict:
            if key == "Final":
                continue

            if baseline_dict[key][0] <= algo_final:
                stoppage = baseline_dict[key][1]
                break

        if stoppage is None:
            if baseline_dict['Final'][0] <= algo_final and not skip:
                stoppage = baseline_dict['Final'][1]
                DP_catch_up.append(stoppage / instance_dict['Final'][1])
                print((Gaps[index], stoppage / instance_dict['Final'][1]))
            else:
                unmatched += 1
                continue
        else:
            if not skip:
                DP_catch_up.append(stoppage / instance_dict['Final'][1])
                print((Gaps[index], stoppage / instance_dict['Final'][1]))

    CI = {}
    x = []
    y = []
    for key in time_obj_map:
        if not time_obj_map[key]:
            continue

        CI[key] = st.t.interval(0.95, len(time_obj_map[key]) - 1, loc=numpy.mean(time_obj_map[key]),
                                scale=st.sem(time_obj_map[key]))

        x.append(key)
        y.append(statistics.mean(time_obj_map[key]))

        if math.isnan(CI[key][0]):
            del CI[key]
            index = x.index(key)
            del x[index]
            del y[index]

    try:
        del CI['Final']
    except:
        print("already deleted")

    print("Unmatched objectives: " + str(unmatched))
    print(len(DP_catch_up))
    print("Average final objective gap is: " + str(statistics.mean(Gaps)))
    print("Avg Nr of CG iterations for method: " + str(statistics.mean(method_CG_iters)))
    print("Avg Nr of CG iterations for baseline: " + str(statistics.mean(baseline_CG_iters)))
    print(" Mean run time for solving PP with method: " + str(statistics.mean(method_PP_time)))
    print(" Mean run time for solving PP with baseline: " + str(statistics.mean(baseline_PP_time)))
    print("Mean Run Time for method: " + str(statistics.mean(method_run_times)))
    print("Objective gaps along iterations:" + str(y))
    print("Run time along iterations: " + str(x))
    print("With confidence interval: " + str(CI))
    print("Scale factor of reduction in run_time:" + str(statistics.mean(DP_catch_up)))

    CI_up = [CI[key][1] for key in CI]
    CI_low = [CI[key][0] for key in CI]

    pp.plot(x, y)
    pp.fill_between(x, CI_low, CI_up, color='b', alpha=.1)
    pp.xlabel("Time (s)")
    pp.ylabel("Objective Gap (%)")
    pp.title("Convergence plot")
    pp.show()

    pp.hist(Gaps)
    pp.title("Objective Gaps Histogram")
    pp.show()

    pp.hist(DP_catch_up)
    pp.title("Time Gaps Histogram")
    pp.show()


def create_price(time_matrix, duals):
    if len(duals) < len(time_matrix):
        duals.insert(0, 0)
    assert duals[0] == 0 and len(duals) == len(time_matrix)
    duals = numpy.array(duals)
    prices = (time_matrix - duals) * -1
    numpy.fill_diagonal(prices, 0)
    return prices


def calculate_compatibility(time_windows, travel_times, service_times):
    n = len(travel_times)
    earliest = numpy.reshape(time_windows[:, 0], (n, 1)) + numpy.reshape(service_times, (n, 1)) \
               + travel_times
    feasibles = earliest - time_windows[:, 1]
    earliest[feasibles > 0] = math.inf
    latest = numpy.reshape(time_windows[:, 1], (n, 1)) + numpy.reshape(service_times, (n, 1)) \
             + travel_times
    latest = numpy.minimum(latest, numpy.reshape(time_windows[:, 1], (1, n)))
    latest[earliest == math.inf] = math.inf

    TC_early = numpy.maximum(earliest, numpy.reshape(time_windows[:, 0], (1, n)))
    TC_late = numpy.maximum(latest, numpy.reshape(time_windows[:, 0], (1, n)))
    numpy.fill_diagonal(TC_early, math.inf)
    numpy.fill_diagonal(TC_late, math.inf)

    return TC_early, TC_late


def populate_null_customers(model_size, X, price_adj):
    num_cols = X.shape[2]
    new_X = torch.zeros((1, model_size + 1, num_cols), dtype=X.dtype)
    new_price_adj = torch.zeros((1, model_size + 1, model_size + 1), dtype=X.dtype) + 2
    new_X[:, :X.shape[1], :] = X
    new_price_adj[:, :price_adj.shape[1], :price_adj.shape[1]] = price_adj

    return new_X, new_price_adj


def reshape_problem(coords, demands, time_windows, duals, service_times, time_matrix, prices, dist=None):
    coords = numpy.copy(coords)
    demands = numpy.copy(demands)
    time_windows = numpy.copy(time_windows)
    duals = duals.copy()
    service_times = numpy.copy(service_times)
    time_matrix = numpy.copy(time_matrix)
    prices = numpy.copy(prices)

    remaining_customers = []
    for x in range(1, len(coords)):
        if coords[x, 0] == math.inf:
            demands[x] = math.inf
            time_windows[x, :] = math.inf
            duals[x] = math.inf
            service_times[x] = math.inf
            time_matrix[x, :] = math.inf
            time_matrix[:, x] = math.inf
            prices[x, :] = math.inf
            prices[:, x] = math.inf
            if dist is not None:
                dist[x, :] = math.inf
                dist[:, x] = math.inf
        else:
            remaining_customers.append(x)

    cus_mapping = {}
    for x in range(len(remaining_customers)):
        cus_mapping[x + 1] = remaining_customers[x]

    coords = coords[coords[:, 0] != math.inf]
    demands = demands[demands[:] != math.inf]
    time_windows = time_windows[time_windows[:, 0] != math.inf]
    duals = [duals[x] for x in range(len(duals)) if duals[x] != math.inf]
    service_times = service_times[service_times[:] != math.inf]
    time_matrix = time_matrix[time_matrix[:, 0] != math.inf]
    mask = (time_matrix == math.inf)
    idx = mask.any(axis=0)
    time_matrix = time_matrix[:, ~idx]
    prices = prices[prices[:, 0] != math.inf]
    mask = (prices == math.inf)
    idx = mask.any(axis=0)
    prices = prices[:, ~idx]
    if dist is not None:
        dist = dist[dist[:, 0] != math.inf]
        mask = (dist == math.inf)
        idx = mask.any(axis=0)
        dist = dist[:, ~idx]
    # print("The problem has been reduced to size: " + str(len(coords) - 1))
    if dist is None:
        return coords, demands, time_windows, duals, service_times, time_matrix, prices, cus_mapping
    else:
        return coords, demands, time_windows, duals, service_times, time_matrix, prices, dist, cus_mapping


def remap_route(route, cus_mapping):
    for x in range(1, len(route) - 1):
        route[x] = cus_mapping[route[x]]
    return route


def initialize_columns(num_customers, truck_capacity, time_matrix, service_times, time_windows, demands):
    unvisited_customers = list(range(1, num_customers + 1))
    solution = []
    current_stop = 0
    current_route = [0]
    remaining_capacity = truck_capacity
    current_time = 0
    while len(unvisited_customers) > 0:
        nearest_customers = numpy.argsort(time_matrix[current_stop, :].copy())
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
    route = numpy.zeros(num_customers)
    for customer in ordered_route:
        if customer != 0:
            route[customer - 1] = 1
    return route


def create_forbidden_edges_list(num_customers, forbidden_edges, compelled_edges):
    forbid_copy = forbidden_edges.copy()
    for edge in compelled_edges:
        forbid_copy += [[x, edge[1]] for x in range(num_customers + 1) if x != edge[0]]
    return forbid_copy


def check_route_feasibility(route, time_matrix, time_windows, service_times, demands_data, truck_capacity):
    if len(route) < 3 or route[0] != 0 or route[-1] != 0:
        return False

    current_time = max(time_matrix[0, route[1]], time_windows[route[1], 0])
    total_capacity = 0

    for i in range(1, len(route)):
        if round(current_time, 3) > time_windows[route[i], 1]:
            # print("Time Window violated")
            # print(route[i])
            return False
        current_time += service_times[route[i]]
        total_capacity += demands_data[route[i]]
        if round(total_capacity, 3) > truck_capacity:
            # print("Truck Capacity Violated")
            # print(route[i])
            return False
        if i < len(route) - 1:
            # travel to next node
            current_time += time_matrix[route[i], route[i + 1]]
            current_time = max(current_time, time_windows[route[i + 1], 0])
    return True


def main():
    method = 'K-Opt'
    num_customers = 1000
    addition = "ULGR 10 Large Scale Instances"
    baseline = 'True Large Scale Instances'
    switch = True

    result_analyzer(method, baseline, num_customers, addition, switch)


if __name__ == "__main__":
    main()
