from instance_generator import Instance_Generator
from column_generation import MasterProblem, Subproblem
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


def RL_solve_relaxed_vrp_with_time_windows(VRP_instance, forbidden_edges, compelled_edges,
                                           initial_routes, initial_costs, initial_orders,
                                           model_params, model_load, red_costs):
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
    reoptimize = True
    max_iter = 5000
    max_time = 60 * 60
    iteration = 0
    cum_time = 0
    results_dict = {}
    start_time = time.time()
    dual_plot = []
    while iteration < max_iter:

        if time.time() - start_time > max_time:
            print("Time Limit Reached")
            break

        master_problem.solve()
        duals = master_problem.retain_duals()
        dual_plot += [x for x in duals if x > 0]

        prices = create_price(time_matrix, duals)
        env_params = {'problem_size': num_customers,
                      'pomo_size': num_customers}
        env = Env(**env_params)
        time_1 = time.time()
        env.declare_problem(coords, demands, time_windows,
                            duals, service_times, time_matrix, prices, vehicle_capacity, 1)

        pp_rl_solver = ESPRCTW_RL_solver(env, model, prices)
        ordered_routes, best_route, best_reward = pp_rl_solver.generate_columns()

        # red_costs.append(best_reward)
        # break

        time_2 = time.time()

        for ordered_route in ordered_routes:
            while ordered_route[-1] == ordered_route[-2]:
                ordered_route.pop()
            '''if not check_route_feasibility(ordered_route, time_matrix, time_windows, service_times, demands,
                                           vehicle_capacity):
                print("Infeasible Route Detected")
                sys.exit(0)'''

        while best_route[-1] == best_route[-2]:
            best_route.pop()

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
                cost = sum(
                    time_matrix[ordered_route[i], ordered_route[i + 1]] for i in range(len(ordered_route) - 1))
                route = convert_ordered_route(ordered_route, num_customers)

                master_problem.add_columns([route], [cost], [ordered_route], forbidden_edges, compelled_edges)
                added_orders.append(ordered_route)
            # SEE code_blocks.py
        else:
            reoptimize = False
            # Optimality has been reached
            print("No columns with negative reduced cost found.")
            break

    if reoptimize:
        master_problem.solve()
    sol, obj = master_problem.extract_solution()
    results_dict["Final"] = (obj, time.time() - start_time)
    routes, costs, orders = master_problem.extract_columns()
    # pp.hist(dual_plot, bins=50)
    print("Average duals: " + str(statistics.mean(dual_plot)))
    # print("Std. Dev. of duals: "+str(statistics.stdev(dual_plot)))
    # pp.show()
    master_problem.__delete__()
    return sol, obj, routes, costs, orders, results_dict


class ESPRCTW_RL_solver(object):
    def __init__(self, env, model, prices):
        self.env = env
        self.model = model
        self.prices = prices

    def train(self, steps):
        pass

    def evaluate(self):
        pass

    def return_real_reward(self, decisions):
        real_rewards = torch.zeros((self.env.batch_size, self.env.pomo_size))
        for x in range(self.env.batch_size):
            for y in range(self.env.pomo_size):
                real_rewards[x, y] = sum(
                    [self.prices[int(decisions[r, x, y]), int(decisions[r + 1, x, y])] for r in
                     range(len(decisions) - 1)])

        return real_rewards * -1

    def generate_columns(self):
        self.model.eval()
        with torch.no_grad():
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        decisions = torch.empty((0, self.env.batch_size, self.env.pomo_size), dtype=torch.float32)
        while not done:
            selected, _ = self.model(state)
            decisions = torch.cat((decisions, selected[None, :, :]), dim=0)
            # shape: (max episode length, batch, pomo)
            state, reward, done = self.env.step(selected)
            # shape: (batch, pomo)

        real_rewards = self.return_real_reward(decisions)

        best_rewards_indexes = real_rewards.argmin(dim=1)[0]

        best_column = torch.tensor(decisions[:, 0, best_rewards_indexes], dtype=torch.int).tolist()
        # negative_reduced_costs = real_rewards < -0.001
        # indices = negative_reduced_costs.nonzero()
        sorted_indices = torch.argsort(real_rewards, dim=1, descending=False)[0]

        promising_columns = []
        for index in sorted_indices:
            if real_rewards[0, index] < -0.001 and len(promising_columns) < 100:
                column = torch.tensor(decisions[:, 0, index], dtype=torch.int)
                column = column.tolist()
                promising_columns.append(column)
            else:
                break

        return promising_columns, best_column, float(real_rewards[0, best_rewards_indexes])


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
    red_costs = []
    # directory = config["Solomon Test Dataset"]
    # for instance in os.listdir(directory):
    for experiment in range(50):
        # file = directory + "/" + instance
        # file = directory + "/" + "C206.txt"
        # print(file)

        # VRP_instance = Instance_Generator(file_path=file, config=config)
        VRP_instance = Instance_Generator(N=num_customers)
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
                                                                                               model_load, red_costs)

        print("solution: " + str(sol))
        print("objective: " + str(obj))
        print("number of columns: " + str(len(orders)))

        results.append(obj)
        performance_dicts.append(results_dict)

    mean_obj = statistics.mean(results)
    std_obj = statistics.stdev(results)
    print("The mean objective value is: " + str(mean_obj))
    print("The std dev. objective is: " + str(std_obj))

    pickle_out = open('RL Results N=' + str(num_customers) + " New Instances", 'wb')
    pickle.dump(performance_dicts, pickle_out)
    pickle_out.close()

    # pp.hist(red_costs)
    # pp.title("Reduced Cost Histogram for POMO-CG")
    # pp.show()


if __name__ == "__main__":
    main()
