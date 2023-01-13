import column_generation as cg
from instance_generator import Instance_Generator
import math


class Branch_and_Bound():

    def __init__(self, VRP_instance):
        num_customers = VRP_instance.N
        time_matrix = VRP_instance.time_matrix
        time_windows = VRP_instance.time_windows
        demands = VRP_instance.demands
        vehicle_capacity = VRP_instance.vehicle_capacity
        time_limit = VRP_instance.time_limit
        service_times = VRP_instance.service_times
        self.max_depth = 50
        self.solve_MIP(vehicle_capacity, time_matrix, demands, time_windows, time_limit, num_customers, service_times)

    def solve_MIP(self, vehicle_capacity, time_matrix, demands, time_windows, time_limit, num_customers, service_times):
        depth = 0
        forbidden_edges = []
        compelled_edges = []

        lb_sol, self.best_lb = cg.solve_relaxed_vrp_with_time_windows(vehicle_capacity, time_matrix, demands,
                                                                      time_windows, time_limit, num_customers,
                                                                      service_times, forbidden_edges, compelled_edges,
                                                                      [], [], [])
        self.best_sol, self.best_ub, fractional_routes = self.generate_upper_bound(lb_sol, time_matrix, num_customers,
                                                                                   forbidden_edges, compelled_edges)
        edge = self.determine_branching_rule(fractional_routes)
        optimal_sol = self.branch(vehicle_capacity, time_matrix, demands,
                                  time_windows, time_limit, num_customers,
                                  service_times, edge, depth, forbidden_edges, compelled_edges)

    def generate_upper_bound(self, lb_sol, time_matrix, num_customers):  ####CONSIDER FORBIDDEN AND COMPELLED EDGES
        ub_sol = []
        customers_covered = []
        potentials = []
        obj = 0
        for entry in lb_sol:
            route = entry[2]
            if entry[1] == 1:
                ub_sol.append(route)
                customers_covered += route[1:-1]
                obj += sum(time_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1))
            else:
                potentials.append(route)
        potentials.sort(key=lambda x: len(x), reverse=True)

        for route in potentials:
            if len(customers_covered) < num_customers:
                route_adj = [route[x] for x in range(len(route)) if route[x] not in customers_covered]
                ub_sol.append(route_adj)
                customers_covered += route_adj[1:-1]
                obj += sum(time_matrix[route_adj[i]][route_adj[i + 1]] for i in range(len(route_adj) - 1))
            else:
                break
        assert len(customers_covered) == num_customers
        return ub_sol, obj, potentials

    def determine_branching_rule(self, fractional_routes):
        for route in fractional_routes:
            for i in range(1, len(route) - 2):
                node_1 = route[i]
                node_2 = route[i + 1]
                for route_2 in fractional_routes:
                    if route != route_2:
                        if node_1 in route_2 and node_2 in route_2:
                            index1 = route_2.index(node_1)
                            index2 = route_2.index(node_2)
                            if abs(index1 - index2) > 1:
                                return [node_1, node_2]

    def branch(self, vehicle_capacity, time_matrix, demands,
               time_windows, time_limit, num_customers,
               service_times, edge, depth, forbidden_edges, compelled_edges):
        if abs(self.best_ub - self.best_lb) / self.best_ub < 0.01 or depth > self.max_depth:
            return self.best_sol, self.best_ub
        depth += 1

        # edge=1
        compelled_copy = compelled_edges.copy()
        compelled_copy.append(edge)
        lb_sol, lb_obj = cg.solve_relaxed_vrp_with_time_windows(vehicle_capacity, time_matrix, demands,
                                                                time_windows, time_limit, num_customers,
                                                                service_times, forbidden_edges, compelled_copy,
                                                                [], [], [])
        if lb_obj > self.best_lb:
            self.best_lb = lb_obj

        ub_sol, ub_obj, fractional_routes = self.generate_upper_bound(lb_sol, time_matrix, num_customers,
                                                                      forbidden_edges, compelled_copy)
        if ub_obj < self.best_ub:
            self.best_ub = ub_obj
            self.best_sol = ub_sol

        if lb_sol < self.best_ub and ub_sol != []:
            edge = self.determine_branching_rule(fractional_routes)
            sol_1, obj_1 = self.branch(vehicle_capacity, time_matrix, demands,
                                       time_windows, time_limit, num_customers,
                                       service_times, edge, depth, forbidden_edges, compelled_copy)
        else:
            sol_1 = ub_sol
            obj_1 = ub_obj

        # edge=0
        forbidden_copy = forbidden_edges.copy()
        forbidden_copy.append(edge)
        lb_sol, lb_obj = cg.solve_relaxed_vrp_with_time_windows(vehicle_capacity, time_matrix, demands,
                                                                time_windows, time_limit, num_customers,
                                                                service_times, forbidden_copy, compelled_edges,
                                                                [], [], [])
        if lb_obj > self.best_lb:
            self.best_lb = lb_obj

        ub_sol, ub_obj, fractional_routes = self.generate_upper_bound(lb_sol, time_matrix, num_customers,
                                                                      forbidden_copy, compelled_edges)
        if ub_obj < self.best_ub:
            self.best_ub = ub_obj
            self.best_sol = ub_sol

        if lb_sol < self.best_ub and ub_sol != []:
            edge = self.determine_branching_rule(fractional_routes)
            sol_2, obj_2 = self.branch(vehicle_capacity, time_matrix, demands,
                                       time_windows, time_limit, num_customers,
                                       service_times, edge, depth, forbidden_copy, compelled_edges)
        else:
            sol_2 = ub_sol
            obj_2 = ub_obj

        if obj_1 < obj_2:
            return sol_1, obj_1
        else:
            return sol_2, obj_2


def main():
    VRP_instance = Instance_Generator(10)
    BNB = Branch_and_Bound(VRP_instance)


if __name__ == "__main__":
    main()
