from column_generation import *

import sys

from instance_generator import Instance_Generator
import time
import pickle
import os

def generate_CVRPTW_data(VRP_instance, forbidden_edges, compelled_edges,
                         initial_routes, initial_costs, initial_orders, coords_list, time_matrix_list,
                         time_windows_list, demands_list, duals_list,
                         vehicle_capacity_list, service_times_list):
    coords = VRP_instance.coords
    time_matrix = VRP_instance.time_matrix
    time_windows = VRP_instance.time_windows
    demands = VRP_instance.demands
    vehicle_capacity = VRP_instance.vehicle_capacity
    time_limit = VRP_instance.time_limit
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

    # Iterate until optimality is reached
    while True:
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
        subproblem = Subproblem(num_customers, vehicle_capacity, time_matrix, demands, time_windows, time_limit,
                                duals, service_times, forbidden_edges)
        ordered_route, reduced_cost = subproblem.solve()
        time_22 = time.time()
        top_labels = sorted(subproblem.top_labels, key=lambda x: x[1])[1:]
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
    return sol, obj, routes, costs, orders


def main():
    num_customers = 20

    coords_list = []
    demands_list = []
    time_matrix_list = []
    time_windows_list = []
    vehicle_capacity_list = []
    duals_list = []
    service_times_list = []

    for x in range(1):
        VRP_instance = Instance_Generator(num_customers)

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

    os.chdir("/gpfs/home6/abdoab/DDOinVRP/Data/")
    pickle_out = open('CVRPTW_data_' + str(num_customers)+"_"+str(time_2), 'wb')
    pickle.dump([coords_list, time_matrix_list, time_windows_list, demands_list, service_times_list,
                 vehicle_capacity_list, duals_list], pickle_out)
    pickle_out.close()


if __name__ == "__main__":
    main()
