import numpy

from instance_generator import Instance_Generator
from column_generation import MasterProblem
from ESPRCTWProblemDef import get_random_problems
import time
from utils import *

import torch
import json
import random
import numpy as np
import sys
import statistics
import argparse
from column_generation import Subproblem
from ESPPRC_heuristic import DSSR_ESPPRC, ESPPRC
from graph_reduction import Node_Reduction
from ESPRCTWEnv import ESPRCTWEnv as Env
from CG_with_RL import ESPRCTW_RL_solver
import matplotlib.pyplot as pp

from scipy.spatial import distance_matrix
from UL_models import GNN
from graph_reduction import Arc_Reduction


def RL_solve_relaxed_vrp_with_time_windows(VRP_instance, forbidden_edges, compelled_edges,
                                           initial_routes, initial_costs, initial_orders, model_path,
                                           reduction_size, heuristic, red_costs):
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

    device = torch.device('cpu')
    GR = GNN(input_dim=9, hidden_dim=64, output_dim=2, n_layers=2)
    checkpoint_main = torch.load(model_path, map_location=device)
    GR.load_state_dict(checkpoint_main)
    temperature = 3.5

    # Iterate until optimality is reached
    max_iter = 5000
    iteration = 0
    cum_time = 0
    results_dict = {}
    start_time = time.time()
    arc_red = False
    reoptimize = True
    max_time = 10 * 60
    while iteration < max_iter:

        if time.time() - start_time > max_time:
            print("Time Limit Reached")
            break
        master_problem.solve()
        duals = master_problem.retain_duals()

        prices = create_price(time_matrix, duals) * -1
        min_val = numpy.min(prices)
        max_val = numpy.max(prices)
        price_scaler = max(abs(max_val), abs(min_val))

        time_1 = time.time()
        tw_scaler = time_windows[0, 1]
        f0 = torch.tensor(np.expand_dims(coords[1:, :], 0), dtype=torch.float32)
        f1 = torch.tensor(np.expand_dims(duals[1:], 0), dtype=torch.float32)
        f2 = torch.tensor(np.expand_dims(time_windows[1:, :], 0), dtype=torch.float32) / tw_scaler
        f3 = torch.tensor(np.expand_dims(demands[1:], 0), dtype=torch.float32) / vehicle_capacity
        f4 = torch.tensor(np.expand_dims(service_times[1:], 0), dtype=torch.float32) / tw_scaler
        f5 = torch.tensor(np.expand_dims(prices[0, 1:], 0), dtype=torch.float32) / price_scaler
        f6 = torch.tensor(np.expand_dims(prices[1:, 0], 0), dtype=torch.float32) / price_scaler

        # full_price_mean = numpy.mean(prices / price_scaler, axis=0)
        # sorted_mean_price_rank = numpy.sort(full_price_mean[1:])[:reduction_size]
        # price_indices = numpy.argsort(full_price_mean[1:])[:reduction_size]
        # print(sorted_mean_price_rank)
        # print(price_indices)

        '''non0 = numpy.array([d for d in f1[0] if d > 0])
        print(max(non0))
        print(min(non0))
        print(len(non0))
        print(statistics.mean(non0))
        print((numpy.sort(non0 * -1) * -1)[:reduction_size])
        print('-----------------')'''

        dims = f1.shape
        f1 = torch.reshape(f1, (dims[0], dims[1], 1))
        f3 = torch.reshape(f3, (dims[0], dims[1], 1))
        f4 = torch.reshape(f4, (dims[0], dims[1], 1))
        f5 = torch.reshape(f5, (dims[0], dims[1], 1))
        f6 = torch.reshape(f6, (dims[0], dims[1], 1))
        X = torch.cat([f0, f1, f2, f3, f4, f5, f6], dim=2)

        distance_m = torch.tensor(np.expand_dims(prices[1:, 1:], 0), dtype=torch.float32) / price_scaler
        adj = torch.exp(-1. * distance_m / temperature)
        output = GR(X, adj)
        probas = torch.sort(output, 1, descending=True)[0]
        # print(probas[0, :reduction_size, 0])
        sorted_indices = output.argsort(dim=1, descending=True)[0, :reduction_size, 0]

        '''tabu = sorted_indices + 1
        tabu2 = numpy.argsort(numpy.array(duals[1:]) * -1)[:reduction_size] + 1
        tab_count = []
        for x in tabu2:
            if x not in tabu:
                tab_count.append(x)
        print(tabu2)
        print(tab_count[:10])
        print("####################")'''

        NR = Node_Reduction(coords, duals)
        red_cor = NR.reduce_by_indices(sorted_indices)

        red_cor, red_dem, red_tws, red_duals, red_sts, red_tms, red_prices, cus_mapping = reshape_problem(red_cor,
                                                                                                          demands,
                                                                                                          time_windows,
                                                                                                          duals,
                                                                                                          service_times,
                                                                                                          time_matrix,
                                                                                                          prices)

        N = len(red_cor) - 1

        AR = Arc_Reduction(red_prices, red_duals)
        red_prices = AR.neighbor_count(red_tws,red_tms,red_sts,red_dem,vehicle_capacity)

        if heuristic == "DP":
            subproblem = Subproblem(N, vehicle_capacity, red_tms, red_dem, red_tws,
                                    red_duals, red_sts, forbidden_edges)
            ordered_route, reduced_cost, top_labels = subproblem.solve_heuristic(arc_red=arc_red)
        else:
            subproblem = ESPPRC(vehicle_capacity, red_dem, red_tws, red_sts, N,
                                red_tms, red_prices)
            top_labels = subproblem.solve()
            if len(top_labels) > 0:
                ordered_route = top_labels[0].path()
                reduced_cost = top_labels[0].cost
                del top_labels[0]
            else:
                ordered_route = []
                reduced_cost = 0

        # red_costs.append(best_reward)
        # break

        time_2 = time.time()

        ordered_route = remap_route(ordered_route, cus_mapping)
        cost = sum(time_matrix[ordered_route[i], ordered_route[i + 1]] for i in range(len(ordered_route) - 1))
        route = convert_ordered_route(ordered_route, num_customers)

        iteration += 1
        obj_val = master_problem.model.objval
        cum_time += time_2 - time_1
        if iteration % 10 == 0:
            print("Iteration: " + str(iteration))
            print("RC is: " + str(reduced_cost))
            print("Best route: " + str(ordered_route))
            print("The total time spent on PP is: " + str(cum_time))
            print("The objective value is: " + str(obj_val))
            print("The number of columns generated is: " + str(len(top_labels) + 1))
            results_dict[iteration] = (obj_val, time.time() - start_time)

        # Check if the candidate column is optimal
        if reduced_cost < -0.001:
            # Add the column to the master problem
            master_problem.add_columns([route], [cost], [ordered_route], forbidden_edges, compelled_edges)
            added_orders.append(ordered_route)
            for x in range(len(top_labels)):
                if heuristic == "DP":
                    label = top_labels[x]
                else:
                    label = top_labels[x].path()
                label = remap_route(label, cus_mapping)
                cost = sum(time_matrix[label[i], label[i + 1]] for i in range(len(label) - 1))
                route = convert_ordered_route(label, num_customers)
                master_problem.add_columns([route], [cost], [label], forbidden_edges, compelled_edges)
                added_orders.append(label)
        elif arc_red and heuristic == "DP":
            arc_red = False
            print("Changed arc red mode.")
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
    torch.manual_seed(10)

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_customers', type=int, default=200)
    parser.add_argument('--reduction_size', type=int, default=100)
    parser.add_argument('--heuristic', type=str, default="DP")
    example_path = 'C:/Users/abdug/Python/UL4CG/PP/Saved_Models/PP_200/scatgnn_layer_2_hid_64_model_100_temp_3.500.pth'
    parser.add_argument('--model_path', type=str, default=example_path)
    args = parser.parse_args()
    num_customers = args.num_customers
    reduction_size = args.reduction_size
    heuristic = args.heuristic

    model_path = args.model_path

    file = "config.json"
    with open(file, 'r') as f:
        config = json.load(f)

    results = []
    performance_dicts = []
    red_costs = []
    for experiment in range(5):
        VRP_instance = Instance_Generator(N=num_customers)
        forbidden_edges = []
        compelled_edges = []
        initial_routes = []
        initial_costs = []
        initial_orders = []

        sol, obj, routes, costs, orders, results_dict = RL_solve_relaxed_vrp_with_time_windows(VRP_instance,
                                                                                               forbidden_edges,
                                                                                               compelled_edges,
                                                                                               initial_routes,
                                                                                               initial_costs,
                                                                                               initial_orders,
                                                                                               model_path,
                                                                                               reduction_size,
                                                                                               heuristic,
                                                                                               red_costs)

        print("solution: " + str(sol))
        print("objective: " + str(obj))
        print("number of columns: " + str(len(orders)))

        results.append(obj)
        performance_dicts.append(results_dict)

    mean_obj = statistics.mean(results)
    std_obj = statistics.stdev(results)
    print("The mean objective value is: " + str(mean_obj))
    print("The std dev. objective is: " + str(std_obj))

    '''pickle_out = open('ULGR Results N=' + str(num_customers), 'wb')
    pickle.dump(performance_dicts, pickle_out)
    pickle_out.close()'''

    # pp.hist(red_costs)
    # pp.title("Reduced Cost Histogram for POMO-CG")
    # pp.show()


if __name__ == "__main__":
    main()
