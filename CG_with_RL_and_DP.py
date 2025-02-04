from instance_generator import Instance_Generator
from column_generation import MasterProblem, Subproblem
from CG_with_RL import ESPRCTW_RL_solver
import time
from utils import *

import torch
import json
import random
import numpy as np
import sys
import statistics
import argparse

import matplotlib.pyplot as pp

from ESPRCTWEnv import ESPRCTWEnv as Env
from ESPRCTWModel import ESPRCTWModel as Model
from graph_reduction import Node_Reduction


def RL_solve_relaxed_vrp_with_time_windows(VRP_instance, forbidden_edges, compelled_edges,
                                           initial_routes, initial_costs, initial_orders,
                                           model_params, model_load):
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

    model = Model(**model_params)
    device = torch.device('cpu')
    checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
    checkpoint = torch.load(checkpoint_fullname, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Iterate until optimality is reached
    max_iter = 5000
    max_time = 60 * 60
    iteration = 0
    cum_time = 0
    results_dict = {}
    start_time = time.time()
    dual_plot = []
    DP_used = False
    while iteration < max_iter:

        if time.time() - start_time > max_time:
            print("Time Limit Reached")
            break

        master_problem.solve()
        duals = master_problem.retain_duals()
        dual_plot += [x for x in duals if x > 0]

        prices = create_price(time_matrix, duals)
        if not DP_used:
            env_params = {'problem_size': num_customers,
                          'pomo_size': num_customers}
            env = Env(**env_params)
            time_1 = time.time()
            env.declare_problem(coords, demands, time_windows,
                                duals, service_times, time_matrix, prices, vehicle_capacity, 1, True)

            pp_rl_solver = ESPRCTW_RL_solver(env, model, prices)
            ordered_routes, best_route, best_reward = pp_rl_solver.generate_columns()
            time_2 = time.time()

            for ordered_route in ordered_routes:
                while ordered_route[-1] == ordered_route[-2]:
                    ordered_route.pop()

            while best_route[-1] == best_route[-2]:
                best_route.pop()


        else:
            prices = prices * -1
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
            time_1 = time.time()
            subproblem = Subproblem(N, vehicle_capacity, red_tms, red_dem, red_tws,
                                    red_duals, red_sts, forbidden_edges, red_prices)
            best_route, best_reward, ordered_routes = subproblem.solve_heuristic(arc_red=False, policy="DP",
                                                                                 max_threads=N)
            time_2 = time.time()

            best_route = remap_route(best_route, cus_mapping)

        iteration += 1
        obj_val = master_problem.model.objval
        cum_time += time_2 - time_1
        if iteration % 10 == 0:
            print("Iteration: " + str(iteration))
            print("RC is " + str(best_reward))
            print("Best route: " + str(best_route))
            print("The total time spent on PP is :" + str(cum_time))
            print("The objective value is: " + str(obj_val))
            print("The number of columns generated is: " + str(len(ordered_routes)))
            results_dict[iteration] = (obj_val, time.time() - start_time)

        if len(ordered_routes) > 0:
            for ordered_route in ordered_routes:
                # Add the column to the master problem
                if DP_used:
                    ordered_route = remap_route(ordered_route, cus_mapping)
                cost = sum(
                    time_matrix[ordered_route[i], ordered_route[i + 1]] for i in range(len(ordered_route) - 1))
                route = convert_ordered_route(ordered_route, num_customers)

                master_problem.add_columns([route], [cost], [ordered_route], forbidden_edges, compelled_edges)
                added_orders.append(ordered_route)
        elif not DP_used:
            DP_used = True
            print("Changed to DP mode.")
        else:
            # Optimality has been reached
            print("No columns with negative reduced cost found.")
            break

    sol, obj = master_problem.extract_solution()
    results_dict["Final"] = (obj, time.time() - start_time)
    routes, costs, orders = master_problem.extract_columns()
    # pp.hist(dual_plot, bins=50)
    print("Average duals: " + str(statistics.mean(dual_plot)))
    # print("Std. Dev. of duals: "+str(statistics.stdev(dual_plot)))
    # pp.show()
    return sol, obj, routes, costs, orders, results_dict


def main():
    random.seed(10)
    np.random.seed(10)
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_customers', type=int, default=100)
    example_path = 'C:/Users/abdug/Python/POMO-implementation/ESPRCTW/POMO/result/model100_scaler1_Nby2'
    parser.add_argument('--model_path', type=str, default=example_path)
    parser.add_argument('--epoch', type=int, default=200)
    args = parser.parse_args()
    num_customers = args.num_customers

    path = args.model_path
    epoch = args.epoch

    file = "config.json"
    with open(file, 'r') as f:
        config = json.load(f)

    results = []
    performance_dicts = []
    directory = config["Solomon Test Dataset"]
    for instance in os.listdir(directory):
        # for experiment in range(50):
        file = directory + "/" + instance
        print(file)

        VRP_instance = Instance_Generator(file_path=file, config=config)
        # VRP_instance = Instance_Generator(N=num_customers)

        forbidden_edges = []
        compelled_edges = []
        initial_routes = []
        initial_costs = []
        initial_orders = []

        model_params = {
            'embedding_dim': 128,
            'sqrt_embedding_dim': 128 ** (1 / 2),
            'encoder_layer_num': 6,
            'qkv_dim': 16,
            'head_num': 8,
            'logit_clipping': 10,
            'ff_hidden_dim': 512,
            'eval_type': 'argmax',
        }

        model_load = {
            'path': path,
            'epoch': epoch}

        sol, obj, routes, costs, orders, results_dict = RL_solve_relaxed_vrp_with_time_windows(VRP_instance,
                                                                                               forbidden_edges,
                                                                                               compelled_edges,
                                                                                               initial_routes,
                                                                                               initial_costs,
                                                                                               initial_orders,
                                                                                               model_params,
                                                                                               model_load)

        print("solution: " + str(sol))
        print("objective: " + str(obj))
        print("number of columns: " + str(len(orders)))

        results.append(obj)
        performance_dicts.append(results_dict)

    mean_obj = statistics.mean(results)
    std_obj = statistics.stdev(results)
    print("The mean objective value is: " + str(mean_obj))
    print("The std dev. objective is: " + str(std_obj))

    pickle_out = open('RL+DP Results N=' + str(num_customers) + " Solomon", 'wb')
    pickle.dump(performance_dicts, pickle_out)
    pickle_out.close()

    # pp.hist(red_costs)
    # pp.title("Reduced Cost Histogram for POMO-CG")
    # pp.show()


if __name__ == "__main__":
    main()
