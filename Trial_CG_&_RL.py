from instance_generator import Instance_Generator
from column_generation import initialize_columns, check_route_feasibility, create_forbidden_edges_list
from column_generation import MasterProblem, convert_ordered_route
import time
from ESPRCTW_RL_trainer import calculate_price

import torch
import json
import random
import numpy as np
import sys
import statistics

import matplotlib.pyplot as pp

sys.path.insert(0, r'C:/Users/abdug/Python/POMO-implementation/ESPRCTW/POMO')
sys.path.insert(0, r'C:/Users/abdug/Python/POMO-implementation/ESPRCTW')
from ESPRCTWEnv import ESPRCTWEnv as Env
from ESPRCTWModel import ESPRCTWModel as Model


def RL_solve_relaxed_vrp_with_time_windows(coords, vehicle_capacity, time_matrix, demands, time_windows,
                                           num_customers, service_times, forbidden_edges, compelled_edges,
                                           initial_routes, initial_costs, initial_orders,
                                           model_params, env_params, model_load):
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

    # Iterate until optimality is reached
    while True:
        master_problem.solve()
        print("The objective value of the RMP is: " + str(master_problem.model.objval))
        duals = master_problem.retain_duals()

        env = Env(**env_params)
        prices = calculate_price(time_matrix, duals)

        env.declare_problem(coords, demands, time_windows,
                            duals, service_times, time_matrix, prices, vehicle_capacity)
        model = Model(**model_params)
        prices = prices * -1

        device = torch.device('cpu')
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        pp_rl_solver = ESPRCTW_RL_solver(env, model, prices)

        ordered_routes, best_route, best_reward = pp_rl_solver.generate_columns()

        for ordered_route in ordered_routes:
            while ordered_route[-1] == ordered_route[-2]:
                ordered_route.pop()
            if not check_route_feasibility(ordered_route, time_matrix, time_windows, service_times, demands,
                                           vehicle_capacity):
                print("Infeasible Route Detected")
                sys.exit(0)

        print("RC is " + str(best_reward))
        print(best_route)
        print("The number of columns generated is: " + str(len(ordered_routes)))

        # Check if the candidate column is optimal
        if len(ordered_routes) > 0:
            for ordered_route in ordered_routes:
                if ordered_route not in added_orders:
                    # Add the column to the master problem
                    cost = sum(
                        time_matrix[ordered_route[i], ordered_route[i + 1]] for i in range(len(ordered_route) - 1))
                    route = convert_ordered_route(ordered_route, num_customers)

                    master_problem.add_columns([route], [cost], [ordered_route], forbidden_edges, compelled_edges)
                    added_orders.append(ordered_route)
        else:
            # Optimality has been reached
            print("No columns with negative reduced cost found.")
            break

    sol, obj = master_problem.extract_solution()
    routes, costs, orders = master_problem.extract_columns()
    return sol, obj, routes, costs, orders


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

        return real_rewards

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

        best_rewards_indexes = real_rewards.argmin(dim=1)
        best_column = []
        for x in range(len(decisions)):
            best_column.append(int(decisions[x, 0, best_rewards_indexes[0]]))

        negative_reduced_costs = real_rewards < -0.0000001
        indices = negative_reduced_costs.nonzero()
        promising_columns = []
        for index in indices:
            column = torch.tensor(decisions[:, index[0], index[1]], dtype=torch.int)
            column = column.tolist()
            promising_columns.append(column)

        return promising_columns, best_column, float(real_rewards[0, best_rewards_indexes[0]])


def main():
    random.seed(10)
    np.random.seed(10)

    file = "config.json"
    with open(file, 'r') as f:
        config = json.load(f)

    results = []
    for experiment in range(1):
        num_customers = 20
        # instance = config["Solomon Dataset"] + "/C101.txt"
        # print("The following instance is used: " + instance)
        print("This instance has " + str(num_customers) + " customers.")
        VRP_instance = Instance_Generator(N=num_customers)
        time_matrix = VRP_instance.time_matrix
        time_windows = VRP_instance.time_windows
        demands = VRP_instance.demands
        coords = VRP_instance.coords
        vehicle_capacity = VRP_instance.vehicle_capacity
        service_times = VRP_instance.service_times
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

        env_params = {'problem_size': num_customers,
                      'pomo_size': 20}

        model_load = {
            'path': 'C:/Users/abdug/Python/POMO-implementation/ESPRCTW/POMO/result/saved_esprctw100_model_heuristic_data_prop_train',
            'epoch': 100}

        time_1 = time.time()
        sol, obj, routes, costs, orders = RL_solve_relaxed_vrp_with_time_windows(coords, vehicle_capacity, time_matrix,
                                                                                 demands,
                                                                                 time_windows,
                                                                                 num_customers, service_times,
                                                                                 forbidden_edges,
                                                                                 compelled_edges,
                                                                                 initial_routes, initial_costs,
                                                                                 initial_orders, model_params,
                                                                                 env_params, model_load)
        time_2 = time.time()

        print("time: " + str(time_2 - time_1))
        print("solution: " + str(sol))
        print("objective: " + str(obj))
        print("number of columns: " + str(len(orders)))

        results.append(obj)

    mean_obj = statistics.mean(results)
    std_obj = statistics.stdev(results)
    print("The mean objective value is: " + str(mean_obj))
    print("The std dev. objective is: " + str(std_obj))

    pp.hist(results)
    pp.show()


if __name__ == "__main__":
    main()
