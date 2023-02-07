import column_generation as cg
from instance_generator import Instance_Generator


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

        lb_sol, self.best_lb, routes, costs, orders = cg.solve_relaxed_vrp_with_time_windows(vehicle_capacity,
                                                                                             time_matrix, demands,
                                                                                             time_windows, time_limit,
                                                                                             num_customers,
                                                                                             service_times,
                                                                                             forbidden_edges,
                                                                                             compelled_edges,
                                                                                             [], [], [])
        self.best_sol, self.best_ub, fractional_routes = self.generate_upper_bound(lb_sol, time_matrix, num_customers)
        edge = self.determine_branching_rule(fractional_routes)
        optimal_sol, optimal_obj = self.branch(vehicle_capacity, time_matrix, demands,
                                               time_windows, time_limit, num_customers,
                                               service_times, edge, depth, forbidden_edges, compelled_edges, routes,
                                               costs, orders)
        print(optimal_sol)
        print(optimal_obj)

    def generate_upper_bound(self, lb_sol, time_matrix, num_customers):  ####CONSIDER FORBIDDEN AND COMPELLED EDGES
        ub_sol = []
        customers_covered = []
        fractional_routes = []
        obj = 0
        for entry in lb_sol:
            route = entry[2]
            if entry[1] == 1:
                ub_sol.append(route)
                customers_covered += route[1:-1]
                obj += sum(time_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1))
            else:
                fractional_routes.append(entry[1:])
        fractional_routes.sort(key=lambda x: x[0], reverse=True)

        for entry in fractional_routes:
            if len(customers_covered) < num_customers:
                route = entry[1]
                route_adj = [route[x] for x in range(len(route)) if route[x] not in customers_covered]
                ub_sol.append(route_adj)
                customers_covered += route_adj[1:-1]
                obj += sum(time_matrix[route_adj[i]][route_adj[i + 1]] for i in range(len(route_adj) - 1))
            else:
                break

        return ub_sol, obj, fractional_routes

    def determine_branching_rule(self, fractional_routes):  # CHANGE BRANCHING STRATEGY

        edge_scores = {}
        for entry in fractional_routes:
            score = entry[0]
            route = entry[1]
            for i in range(len(route) - 1):
                edge = (route[i], route[i + 1])
                try:
                    edge_scores[edge] += score
                except:
                    edge_scores[edge] = score
        edge_scores_list = [(edge, score) for edge, score in edge_scores.items() if edge_scores[edge] < 1]
        edge_scores_list.sort(key=lambda x: x[1], reverse=True)
        return list(edge_scores_list[0][0])

    def branch(self, vehicle_capacity, time_matrix, demands,
               time_windows, time_limit, num_customers,
               service_times, edge, depth, forbidden_edges, compelled_edges, routes, costs, orders):

        if (self.best_ub - self.best_lb) / self.best_ub < 0.001 or depth > self.max_depth:
            return self.best_sol, self.best_ub
        depth += 1

        # edge=1
        compelled_copy = compelled_edges.copy()
        compelled_copy.append(edge)
        lb_sol1, lb_obj1, routes1, costs1, orders1 = cg.solve_relaxed_vrp_with_time_windows(vehicle_capacity,
                                                                                            time_matrix,
                                                                                            demands,
                                                                                            time_windows, time_limit,
                                                                                            num_customers,
                                                                                            service_times,
                                                                                            forbidden_edges,
                                                                                            compelled_copy,
                                                                                            routes, costs, orders)
        if lb_obj1 > self.best_lb:
            self.best_lb = lb_obj1

        ub_sol1, ub_obj1, fractional_routes1 = self.generate_upper_bound(lb_sol1, time_matrix, num_customers)
        if ub_obj1 < self.best_ub:
            self.best_ub = ub_obj1
            self.best_sol = ub_sol1

        # edge=0
        forbidden_copy = forbidden_edges.copy()
        forbidden_copy.append(edge)
        lb_sol2, lb_obj2, routes2, costs2, orders2 = cg.solve_relaxed_vrp_with_time_windows(vehicle_capacity,
                                                                                            time_matrix,
                                                                                            demands,
                                                                                            time_windows, time_limit,
                                                                                            num_customers,
                                                                                            service_times,
                                                                                            forbidden_copy,
                                                                                            compelled_edges,
                                                                                            routes, costs, orders)
        if lb_obj2 > self.best_lb:
            self.best_lb = lb_obj2

        ub_sol2, ub_obj2, fractional_routes2 = self.generate_upper_bound(lb_sol2, time_matrix, num_customers)
        if ub_obj2 < self.best_ub:
            self.best_ub = ub_obj2
            self.best_sol = ub_sol2

        if lb_obj1 < self.best_ub and ub_sol1 != []:  #
            edge = self.determine_branching_rule(fractional_routes1)

            sol_1, obj_1 = self.branch(vehicle_capacity, time_matrix, demands,
                                       time_windows, time_limit, num_customers,
                                       service_times, edge, depth, forbidden_edges, compelled_copy, routes1, costs1,
                                       orders1)
        else:
            sol_1 = ub_sol1
            obj_1 = ub_obj1
            print("Node Pruned 1")

        if lb_obj2 < self.best_ub and ub_sol2 != []:
            edge = self.determine_branching_rule(fractional_routes2)
            sol_2, obj_2 = self.branch(vehicle_capacity, time_matrix, demands,
                                       time_windows, time_limit, num_customers,
                                       service_times, edge, depth, forbidden_edges, compelled_edges, routes2, costs2,
                                       orders2)
        else:
            sol_2 = ub_sol2
            obj_2 = ub_obj2
            print("Node Pruned 2")

        if obj_1 < obj_2:
            return sol_1, obj_1
        else:
            return sol_2, obj_2


def main():
    num_customers = 10
    VRP_instance = Instance_Generator(num_customers)
    BNB = Branch_and_Bound(VRP_instance)


if __name__ == "__main__":
    main()
