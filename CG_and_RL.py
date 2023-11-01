from instance_generator import Instance_Generator
from column_generation import initialize_columns, check_route_feasibility, create_forbidden_edges_list
from column_generation import MasterProblem, convert_ordered_route
import time
import sys
import pickle

from ESPRCTW_RL_solver import ESPRCTW_Env, mask_fn
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy

import random
import numpy as np


def RL_solve_relaxed_vrp_with_time_windows(vehicle_capacity, time_matrix, demands, time_windows, time_limit,
                                           num_customers,
                                           service_times, forbidden_edges, compelled_edges,
                                           initial_routes, initial_costs, initial_orders, solve):
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
    master_problem.solve()
    duals = master_problem.retain_duals()

    env = ESPRCTW_Env(num_customers, vehicle_capacity, time_matrix, demands, time_windows, time_limit, duals,
                      service_times, forbidden_edges)
    env = ActionMasker(env, mask_fn)

    if not solve:
        model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1)
    else:
        model = MaskablePPO.load('RL solver for ESPRCTW -' + str(num_customers))
        # pickle_in = open('RL solver for ESPRCTW -'+str(num_customers), 'rb')
        # pp_rl_solver = pickle.load(pickle_in)

    pp_rl_solver = ESPRCTW_RL_solver(env, model)

    steps = 1000
    # Iterate until optimality is reached
    while True:
        time_11 = time.time()
        if not solve:
            pp_rl_solver.train(steps)
        column_dict = pp_rl_solver.generate_columns(True, 1)
        ordered_route, reduced_cost = column_dict[1]
        reduced_cost = reduced_cost * -1
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
            master_problem.solve()
            duals = master_problem.retain_duals()
            pp_rl_solver.update_problem(num_customers, vehicle_capacity, time_matrix, demands, time_windows,
                                        time_limit, duals, service_times, forbidden_edges)
        else:
            # Optimality has been reached
            print("Addition Failed")
            if not solve:
                pp_rl_solver.model.save('RL solver for ESPRCTW -' + str(num_customers))
                # pickle_out = open('RL solver for ESPRCTW -'+str(num_customers), 'wb')
                # pickle.dump(pp_rl_solver, pickle_out)
                # pickle_out.close()
            break

    sol, obj = master_problem.extract_solution()
    routes, costs, orders = master_problem.extract_columns()
    return sol, obj, routes, costs, orders


class ESPRCTW_RL_solver(object):
    def __init__(self, env, model):
        self.env = env  # ActionMasker(env, mask_fn)

        self.model = model  # MaskablePPO(MaskableActorCriticPolicy, self.env, verbose=1)

    def train(self, steps):
        self.model.learn(total_timesteps=steps, log_interval=1000)

    def evaluate(self):
        vec_env = DummyVecEnv([lambda: self.env])  # self.model.get_env()
        return evaluate_policy(self.model, vec_env, deterministic=True)

    def update_problem(self, num_customers, vehicle_capacity, time_matrix, demands, time_windows,
                       time_limit, duals, service_times, forbidden_edges):

        self.env.unwrapped.update_instance(num_customers, vehicle_capacity, time_matrix, demands, time_windows,
                                           time_limit, duals, service_times, forbidden_edges)

    def generate_columns(self, deterministic, column_number):
        if deterministic:
            column_number = 1

        label_dict = {}
        vec_env = DummyVecEnv([lambda: self.env])  # self.model.get_env()
        for i in range(1, column_number + 1):
            label = [0]
            obs = vec_env.reset()
            done = False
            while not done:
                action_mask = self.env.unwrapped.valid_action_mask()
                action, _state = self.model.predict(obs, action_masks=action_mask, deterministic=deterministic)
                label.append(int(action))
                obs, reward, done, info = vec_env.step(action)

            reward = self.env.unwrapped.calculate_real_reward(label)
            label_dict[i] = (label, reward)

        return label_dict


def main():
    random.seed(5)
    np.random.seed(25)
    num_customers = 20
    print("This instance has " + str(num_customers) + " customers.")
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
    solve = False
    sol, obj, routes, costs, orders = RL_solve_relaxed_vrp_with_time_windows(vehicle_capacity, time_matrix, demands,
                                                                             time_windows, time_limit,
                                                                             num_customers, service_times,
                                                                             forbidden_edges,
                                                                             compelled_edges,
                                                                             initial_routes, initial_costs,
                                                                             initial_orders, solve)
    time_2 = time.time()

    print("time: " + str(time_2 - time_1))
    print("solution: " + str(sol))
    print("objective: " + str(obj))
    print("number of columns: " + str(len(orders)))


if __name__ == "__main__":
    main()
