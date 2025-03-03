import math

import numpy

from instance_generator import Instance_Generator
from column_generation import MasterProblem
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


def UL_solve_relaxed_vrp_with_time_windows(VRP_instance, forbidden_edges, compelled_edges,
                                           initial_routes, initial_costs, initial_orders, model_path,
                                           model_size, red_param, red_costs):
    coords = VRP_instance.coords
    time_matrix = VRP_instance.time_matrix
    time_windows = VRP_instance.time_windows
    demands = VRP_instance.demands
    vehicle_capacity = VRP_instance.vehicle_capacity
    service_times = VRP_instance.service_times
    num_customers = VRP_instance.N

    # Ensure all input lists are of the same length
    assert len(time_matrix) == len(demands) == len(time_windows)
    assert num_customers <= model_size

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
    GR = GNN(input_dim=7, hidden_dim=64, output_dim=model_size + 1, n_layers=2)
    checkpoint_main = torch.load(model_path, map_location=device)
    GR.load_state_dict(checkpoint_main)
    temperature = 3.5

    tw_scaler = time_windows[0, 1]

    TC = calculate_compatibility(time_windows, time_matrix, service_times)[1]
    TC = torch.tensor(TC, dtype=torch.float32) / tw_scaler

    f0 = torch.tensor(np.expand_dims(coords, 0), dtype=torch.float32)
    f2 = torch.tensor(np.expand_dims(time_windows, 0), dtype=torch.float32) / tw_scaler
    f3 = torch.tensor(np.expand_dims(demands, 0), dtype=torch.float32) / vehicle_capacity
    f4 = torch.tensor(np.expand_dims(service_times, 0), dtype=torch.float32) / tw_scaler

    dims = f3.shape
    f3 = torch.reshape(f3, (dims[0], dims[1], 1))
    f4 = torch.reshape(f4, (dims[0], dims[1], 1))

    mask = torch.ones(model_size + 1, model_size + 1).cpu()
    mask.fill_diagonal_(0)

    # Iterate until optimality is reached
    max_iter = 5000
    iteration = 0
    cum_time = 0
    results_dict = {}
    start_time = time.time()
    reoptimize = True
    max_time = 3 * 60
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

        f1 = torch.tensor(np.expand_dims(duals, 0), dtype=torch.float32)
        f1 = torch.reshape(f1, (dims[0], dims[1], 1))
        X = torch.cat([f0, f1, f2, f3, f4], dim=2)

        p_scaled = torch.tensor(prices, dtype=torch.float32) / price_scaler
        price_adj = torch.zeros(p_scaled.shape)
        disc_price_neg = p_scaled * torch.exp(-1 * TC - torch.reshape(f3[0], (1, len(f3[0]))))
        price_adj[p_scaled < 0] = disc_price_neg[p_scaled < 0]
        disc_price_pos = p_scaled * torch.exp(TC + torch.reshape(f3[0], (1, len(f3[0]))))
        price_adj[p_scaled > 0] = disc_price_pos[p_scaled > 0]
        price_adj[TC == math.inf] = 2

        distance_m = price_adj.unsqueeze(0)
        if model_size != num_customers:
            X, distance_m = populate_null_customers(model_size, X, price_adj)

        adj = torch.exp(-1. * distance_m / temperature)
        adj *= mask
        output = GR(X, adj)
        if num_customers != model_size:
            output = output[:num_customers + 1, :num_customers + 1]
        point_wise_distance = torch.matmul(output, torch.roll(torch.transpose(output, 1, 2), -1, 1))[0]

        AR = Arc_Reduction(prices, duals)
        red_prices, dist = AR.ml_arc_reduction(point_wise_distance, m=red_param, price_adj_mat=price_adj)

        NR = Node_Reduction(coords, duals)
        red_cor = NR.dual_based_elimination()
        red_cor, red_dem, red_tws, red_duals, red_sts, red_tms, red_prices2, red_dist, cus_mapping = \
            reshape_problem(red_cor, demands, time_windows, duals, service_times, time_matrix, prices, dist)

        N = len(red_cor) - 1
        subproblem = Subproblem(N, vehicle_capacity, red_tms, red_dem, red_tws,
                                red_duals, red_sts, forbidden_edges, red_prices2)
        ordered_route, reduced_cost, top_labels = subproblem.solve_heuristic(policy="k-opt", dist=red_dist,
                                                                             k_opt_iter=20, max_threads=100)
        # red_costs.append(reduced_cost)
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
                label = top_labels[x]
                label = remap_route(label, cus_mapping)
                cost = sum(time_matrix[label[i], label[i + 1]] for i in range(len(label) - 1))
                route = convert_ordered_route(label, num_customers)
                master_problem.add_columns([route], [cost], [label], forbidden_edges, compelled_edges)
                added_orders.append(label)
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
    parser.add_argument('--num_customers', type=int, default=400)
    parser.add_argument('--red_param', type=int, default=10)
    example_path = 'C:/Users/abdug/Python/UL4CG/PP/Saved_Models/PP_500/scatgnn_layer_2_hid_64_model_300_temp_3.500.pth'
    parser.add_argument('--model_path', type=str, default=example_path)
    parser.add_argument('--model_size', type=int, default=500)
    args = parser.parse_args()
    num_customers = args.num_customers
    red_param = args.red_param
    model_size = args.model_size
    model_path = args.model_path

    file = "config.json"
    with open(file, 'r') as f:
        config = json.load(f)

    print("Unsupervised Learning Module Used")
    print("Instances of size: " + str(num_customers))
    results = []
    performance_dicts = []
    red_costs = []
    directory = config["G&H Dataset"] + str(num_customers)
    for instance in os.listdir(directory):
        # for experiment in range(50):
        file = directory + "/" + instance
        # file = directory + "/" + "C206.txt"
        print(file)

        # VRP_instance = Instance_Generator(N=num_customers)
        VRP_instance = Instance_Generator(file_path=file, config=config, instance_type="G&H")

        forbidden_edges = []
        compelled_edges = []
        initial_routes = []
        initial_costs = []
        initial_orders = []

        sol, obj, routes, costs, orders, results_dict = UL_solve_relaxed_vrp_with_time_windows(VRP_instance,
                                                                                               forbidden_edges,
                                                                                               compelled_edges,
                                                                                               initial_routes,
                                                                                               initial_costs,
                                                                                               initial_orders,
                                                                                               model_path,
                                                                                               model_size,
                                                                                               red_param,
                                                                                               red_costs)

        # print("solution: " + str(sol))
        print("objective: " + str(obj))
        print("number of columns: " + str(len(orders)))

        results.append(obj)
        performance_dicts.append(results_dict)

    mean_obj = statistics.mean(results)
    std_obj = statistics.stdev(results)
    print("The mean objective value is: " + str(mean_obj))
    print("The std dev. objective is: " + str(std_obj))

    pickle_out = open('K-Opt Results N=' + str(num_customers) + ' ULGR ' + str(red_param), 'wb')
    pickle.dump(performance_dicts, pickle_out)
    pickle_out.close()

    '''pp.hist(red_costs, bins=10)
    pp.title("Reduced Cost Histogram for ULGR")
    pp.show()'''


if __name__ == "__main__":
    main()
