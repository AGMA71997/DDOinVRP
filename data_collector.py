from utils import *
from column_generation import MasterProblem, Subproblem
import sys
import json
import threading

from instance_generator import Instance_Generator
import time
import pickle
import os
import argparse


def generate_CVRPTW_data(VRP_instance, forbidden_edges, compelled_edges,
                         initial_routes, initial_costs, initial_orders, coords_list, time_matrix_list,
                         time_windows_list, demands_list, duals_list,
                         vehicle_capacity_list, service_times_list):
    coords = VRP_instance.coords
    time_matrix = VRP_instance.time_matrix
    time_windows = VRP_instance.time_windows
    demands = VRP_instance.demands
    vehicle_capacity = VRP_instance.vehicle_capacity
    service_times = VRP_instance.service_times
    num_customers = VRP_instance.N

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
    max_iter = 500
    iteration = 0
    # Iterate until optimality is reached
    try:
        while iteration < max_iter:
            master_problem.solve()
            duals = master_problem.retain_duals()

            coords_list.append(coords)
            time_matrix_list.append(time_matrix)
            time_windows_list.append(time_windows)
            demands_list.append(demands)
            vehicle_capacity_list.append(vehicle_capacity)
            service_times_list.append(service_times)
            duals_list.append(duals)

            # Consider saving problem parameters here in pickle files for comparison.
            time_11 = time.time()
            subproblem = Subproblem(num_customers, vehicle_capacity, time_matrix, demands, time_windows,
                                    duals, service_times, forbidden_edges)
            if heuristic:
                ordered_route, reduced_cost, top_labels = subproblem.solve_heuristic()
            else:
                ordered_route, reduced_cost, top_labels = subproblem.solve()
            time_22 = time.time()

            # subproblem.render_solution(ordered_route)
            cost = sum(time_matrix[ordered_route[i], ordered_route[i + 1]] for i in range(len(ordered_route) - 1))
            route = convert_ordered_route(ordered_route, num_customers)
            iteration += 1
            if iteration % 100 == 0:
                print("Iteration: " + str(iteration))
                # print("RC is " + str(reduced_cost))
                print("Total solving time for PP is: " + str(time_22 - time_11))
                print(ordered_route)
                print("The objective value is: " + str(master_problem.model.objval))

            # Check if the candidate column is optimal
            if reduced_cost < 0 and ordered_route not in added_orders:
                # Add the column to the master problem
                master_problem.add_columns([route], [cost], [ordered_route], forbidden_edges, compelled_edges)
                added_orders.append(ordered_route)
                # print("Another " + str(len(top_labels)) + " are added.")
                for x in range(len(top_labels)):
                    label = top_labels[x][0]
                    cost = sum(time_matrix[label[i], label[i + 1]] for i in range(len(label) - 1))
                    route = convert_ordered_route(label, num_customers)
                    master_problem.add_columns([route], [cost], [label], forbidden_edges, compelled_edges)
                    added_orders.append(label)
            else:
                # Optimality has been reached
                print("Addition Failed")
                break

        sol, obj = master_problem.extract_solution()
        routes, costs, orders = master_problem.extract_columns()
        master_problem.__delete__()
        return sol, obj, routes, costs, orders
    except:
        print("Loop terminated unexpectedly")
        master_problem.__delete__()
        return [], None, [], [], []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_sequence', type=str, default="")
    parser.add_argument('--num_customers', type=int, default=100)
    parser.add_argument('--heuristic', type=bool, default=True)
    args = parser.parse_args()
    num_customers = args.num_customers

    global heuristic
    heuristic = args.heuristic

    file = "config.json"
    with open(file, 'r') as f:
        config = json.load(f)

    coords_list = []
    demands_list = []
    time_matrix_list = []
    time_windows_list = []
    vehicle_capacity_list = []
    duals_list = []
    service_times_list = []

    directory = config["Solomon Training Dataset"]
    for instance in os.listdir(directory):
        if instance.startswith(args.file_sequence):
            file = directory + "/" + instance
            VRP_instance = Instance_Generator(file_path=file, config=config)

            forbidden_edges = []
            compelled_edges = []
            initial_routes = []
            initial_costs = []
            initial_orders = []
            time_1 = time.time()

            sol, obj, routes, costs, orders = generate_CVRPTW_data(VRP_instance,
                                                                   forbidden_edges,
                                                                   compelled_edges,
                                                                   initial_routes, initial_costs,
                                                                   initial_orders, coords_list,
                                                                   time_matrix_list,
                                                                   time_windows_list, demands_list,
                                                                   duals_list,
                                                                   vehicle_capacity_list, service_times_list)
            time_2 = time.time()

            print("time: " + str(time_2 - time_1))
            print("solution: " + str(sol))
            print("objective: " + str(obj))
            print("number of columns: " + str(len(orders)))

            if not sol:
                break

    if heuristic:
        os.chdir(config["storge_directory_raw_heuristic"] + "/" + str(num_customers))
    else:
        os.chdir(config["storge_directory_raw"] + "/" + str(num_customers))
    pickle_out = open('SAMPLE_ESPRCTW_instances_' + str(num_customers) + "_Solomon_" + str(time_2), 'wb')
    pickle.dump([coords_list, time_matrix_list, time_windows_list, demands_list, service_times_list,
                 vehicle_capacity_list, duals_list], pickle_out)
    pickle_out.close()


if __name__ == "__main__":
    main()
