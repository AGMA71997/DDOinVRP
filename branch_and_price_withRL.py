import time
import column_generation as cg
import CG_and_RL as cgrl
from branch_and_price import generate_upper_bound, determine_branching_rule
from instance_generator import Instance_Generator
import sys
import numpy
import random


class Branch_and_Bound_RL(object):

    def __init__(self, VRP_instance):
        num_customers = VRP_instance.N
        time_matrix = VRP_instance.time_matrix
        time_windows = VRP_instance.time_windows
        demands = VRP_instance.demands
        vehicle_capacity = VRP_instance.vehicle_capacity
        time_limit = VRP_instance.time_limit
        service_times = VRP_instance.service_times
        self.max_depth = 500
        self.solve_MIP(vehicle_capacity, time_matrix, demands, time_windows, time_limit, num_customers, service_times)

    def solve_MIP(self, vehicle_capacity, time_matrix, demands, time_windows, time_limit, num_customers, service_times):
        depth = 0
        forbidden_edges = []
        compelled_edges = []

        initial_routes, initial_costs, initial_orders = cg.initialize_columns(num_customers, vehicle_capacity,
                                                                              time_matrix,
                                                                              service_times, time_windows,
                                                                              demands)
        total_cost = sum(initial_costs[x] for x in range(len(initial_costs)))
        print("The initial routes: " + str(initial_orders))
        print("with total cost: " + str(total_cost))
        for route in initial_orders:
            if not cg.check_route_feasibility(route, time_matrix, time_windows, service_times, demands,
                                              vehicle_capacity):
                print("Infeasible route detected")
                sys.exit(0)
        self.best_ub = total_cost
        self.best_sol = initial_orders
        solve = True

        time_1 = time.time()
        lb_sol, lb_obj, routes, costs, orders = cgrl.RL_solve_relaxed_vrp_with_time_windows(vehicle_capacity,
                                                                                            time_matrix,
                                                                                            demands, time_windows,
                                                                                            time_limit, num_customers,
                                                                                            service_times,
                                                                                            forbidden_edges,
                                                                                            compelled_edges,
                                                                                            initial_routes,
                                                                                            initial_costs,
                                                                                            initial_orders, solve)
        print("best_lb: " + str(lb_obj))
        print("lb_sol: " + str(lb_sol))
        sol, ub, fractional_routes = generate_upper_bound(lb_sol, time_matrix, num_customers)
        if ub < self.best_ub:
            self.best_ub = ub
            self.best_sol = sol
        print("best ub: " + str(self.best_ub))
        print("ub_sol: " + str(self.best_sol))

        edge = determine_branching_rule(fractional_routes)
        optimal_sol, optimal_obj, routes, costs, orders = self.branch(vehicle_capacity, time_matrix, demands,
                                                                      time_windows, time_limit, num_customers,
                                                                      service_times, edge, depth, forbidden_edges,
                                                                      compelled_edges, routes,
                                                                      costs, orders, solve)
        time_2 = time.time()
        print("Optimal sol: " + str(optimal_sol))
        print("Optimal obj: " + str(optimal_obj))
        print("Total routes generated: " + str(len(orders)))
        print("Total time: " + str(time_2 - time_1))

    ##consider compelled and forbidden edges to improve bound generation

    def branch(self, vehicle_capacity, time_matrix, demands, time_windows, time_limit, num_customers,
               service_times, edge, depth, forbidden_edges, compelled_edges, routes, costs, orders,
               solve):

        if depth > self.max_depth:
            print("Maximum Depth Reached")
            return self.best_sol, self.best_ub
        depth += 1
        print("---------")
        print(self.best_ub)
        print(edge)
        print(depth)
        print(compelled_edges)
        print(forbidden_edges)
        print("---------")
        # edge=1
        compelled_edges.append(edge)
        print("Solve for 1")
        lb_sol1, lb_obj1, routes, costs, orders = cgrl.RL_solve_relaxed_vrp_with_time_windows(vehicle_capacity,
                                                                                              time_matrix,
                                                                                              demands,
                                                                                              time_windows, time_limit,
                                                                                              num_customers,
                                                                                              service_times,
                                                                                              forbidden_edges[:],
                                                                                              compelled_edges[:],
                                                                                              routes, costs, orders,
                                                                                              solve)

        ub_sol1, ub_obj1, fractional_routes1 = generate_upper_bound(lb_sol1, time_matrix, num_customers)
        if ub_obj1 < self.best_ub:
            self.best_ub = ub_obj1
            self.best_sol = ub_sol1

        # edge=0
        forbidden_edges.append(edge)
        print("Solve for 0")
        lb_sol2, lb_obj2, routes, costs, orders = cgrl.RL_solve_relaxed_vrp_with_time_windows(vehicle_capacity,
                                                                                              time_matrix,
                                                                                              demands,
                                                                                              time_windows, time_limit,
                                                                                              num_customers,
                                                                                              service_times,
                                                                                              forbidden_edges[:],
                                                                                              compelled_edges[0:-1],
                                                                                              routes, costs, orders,
                                                                                              solve)

        ub_sol2, ub_obj2, fractional_routes2 = generate_upper_bound(lb_sol2, time_matrix, num_customers)
        if ub_obj2 < self.best_ub:
            self.best_ub = ub_obj2
            self.best_sol = ub_sol2

        print("lb_1: " + str(lb_obj1) + " at depth " + str(depth))
        print("ub_1: " + str(ub_obj1))
        if lb_obj1 < self.best_ub and ub_sol1 != []:
            print(str(edge) + "==1 branch")
            edge1 = determine_branching_rule(fractional_routes1)
            if edge1 in forbidden_edges or edge1 in compelled_edges:
                print("edge already added")
                sys.exit()
            sol_1, obj_1, routes, costs, orders = self.branch(vehicle_capacity, time_matrix, demands,
                                                              time_windows, time_limit, num_customers,
                                                              service_times, edge1, depth, forbidden_edges[0:-1],
                                                              compelled_edges[:], routes, costs,
                                                              orders, solve)
        else:
            sol_1 = ub_sol1
            obj_1 = ub_obj1
            print("Node Pruned: " + str(edge) + "==1")

        print("lb_2: " + str(lb_obj2) + " at depth " + str(depth))
        print("ub_2: " + str(ub_obj2))
        if lb_obj2 < self.best_ub and ub_sol2 != []:
            print(str(edge) + "==0 branch")
            edge2 = determine_branching_rule(fractional_routes2)
            if edge2 in forbidden_edges or edge2 in compelled_edges:
                print("edge already added")
                sys.exit()
            sol_2, obj_2, routes, costs, orders = self.branch(vehicle_capacity, time_matrix, demands,
                                                              time_windows, time_limit, num_customers,
                                                              service_times, edge2, depth, forbidden_edges[:],
                                                              compelled_edges[0:-1], routes, costs,
                                                              orders, solve)
        else:
            sol_2 = ub_sol2
            obj_2 = ub_obj2
            print("Node Pruned: " + str(edge) + "==0")

        forbidden_edges.pop()
        compelled_edges.pop()
        if obj_1 < obj_2:
            return sol_1, obj_1, routes, costs, orders
        else:
            return sol_2, obj_2, routes, costs, orders


def main():
    random.seed(5)
    numpy.random.seed(25)
    num_customers = 25
    VRP_instance = Instance_Generator(num_customers)
    BNB = Branch_and_Bound_RL(VRP_instance)


if __name__ == "__main__":
    main()
