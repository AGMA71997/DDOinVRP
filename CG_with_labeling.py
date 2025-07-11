import numpy as np
from instance_generator import Instance_Generator
import random
import time
from column_generation import MasterProblem

from utils import *
import argparse
import json

import matplotlib.pyplot as pp
from graph_reduction import Node_Reduction, Arc_Reduction
from ESPPRC_heuristic import DSSR_ESPPRC

def solve_relaxed_vrp_with_time_windows(VRP_instance, forbidden_edges, compelled_edges, initial_routes,
                                        initial_costs, initial_orders):
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
    arc_red = False
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

        subproblem = DSSR_ESPPRC(vehicle_capacity, red_dem, red_tws, red_sts, N,
                                 red_tms, red_prices)
        top_labels = subproblem.solve()

        if len(top_labels) > 0:
            ordered_route = top_labels[0].path()
            reduced_cost = top_labels[0].cost
            del top_labels[0]
        else:
            ordered_route = []
            reduced_cost = 0

        time_22 = time.time()

        ordered_route = remap_route(ordered_route, cus_mapping)

        cost = sum(time_matrix[ordered_route[i], ordered_route[i + 1]] for i in range(len(ordered_route) - 1))
        route = convert_ordered_route(ordered_route, num_customers)

        iteration += 1
        obj_val = master_problem.model.objval
        cum_time += time_22 - time_11
        if iteration % 10 == 0:
            print("Iteration: " + str(iteration))
            print("Solving time for PP is: " + str(time_22 - time_11))
            print("RC is " + str(reduced_cost))
            print("Best route: " + str(ordered_route))
            print("The objective value is: " + str(obj_val))
            print("The total number of generated columns is: " + str(len(top_labels) + 1))
            print("The total time spent on PP is: " + str(cum_time))
            print('')
            results_dict[iteration] = (obj_val, time.time() - start_time)

        # Check if the candidate column is optimal
        if reduced_cost < -0.001:
            # Add the column to the master problem
            master_problem.add_columns([route], [cost], [ordered_route], forbidden_edges, compelled_edges)
            added_orders.append(ordered_route)
            for x in range(len(top_labels)):
                label = top_labels[x].path()
                rc = top_labels[x].cost
                if rc > -0.001:
                    continue
                label = remap_route(label, cus_mapping)
                cost = sum(time_matrix[label[i], label[i + 1]] for i in range(len(label) - 1))
                route = convert_ordered_route(label, num_customers)
                master_problem.add_columns([route], [cost], [label], forbidden_edges, compelled_edges)
                added_orders.append(label)
        elif arc_red:
            arc_red = False
            print("changed arc red mode.")
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


def main():
    random.seed(10)
    np.random.seed(10)
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_customers', type=int, default=100)
    args = parser.parse_args()
    num_customers = args.num_customers

    file = "config.json"
    with open(file, 'r') as f:
        config = json.load(f)

    results = []
    performance_dicts = []
    for experiment in range(10):
        # instance = config["Solomon Test Dataset"] + "/RC208.txt"
        # print("The following instance is used: " + instance)
        VRP_instance = Instance_Generator(N=num_customers)
        # VRP_instance = Instance_Generator(file_path=instance, config=config)
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
                                                                                            initial_orders)

        print("solution: " + str(sol))
        print("objective: " + str(obj))
        print("number of columns: " + str(len(orders)))

        results.append(obj)
        performance_dicts.append(results_dict)

    mean_obj = statistics.mean(results)
    std_obj = statistics.stdev(results)
    print("The mean objective value is: " + str(mean_obj))
    print("The std dev. objective is: " + str(std_obj))

    pickle_out = open('Labeling Algo Results N=' + str(num_customers), 'wb')
    pickle.dump(performance_dicts, pickle_out)
    pickle_out.close()


if __name__ == "__main__":
    import cProfile

    # cProfile.run('main()')
    main()
