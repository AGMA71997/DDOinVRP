import copy
import sys


from Archive.branch_and_price import generate_upper_bound_2, generate_upper_bound
from instance_generator import Instance_Generator
import time
from utils import *

from sklearn.cluster import SpectralClustering

from Archive.branch_and_price import determine_branching_rule


import json
import random
import numpy as np
import statistics
import argparse
from CG_with_RL import CG_with_RL

use_cuda = torch.cuda.is_available()
if use_cuda:
    cuda_device_num = 0
    torch.cuda.set_device(cuda_device_num)
    device = torch.device('cuda', cuda_device_num)
else:
    device = torch.device("cpu")

def partion_problem(instance, class_predictions):
    return [instance]

def determine_arc_distribution(edge_scores,num_customers):
    dist = torch.zeros((num_customers+1,num_customers+1))
    for entry in edge_scores:
        edge = entry[0]
        score = entry[1]
        assert score <=1
        dist[edge] = score
    return dist[1:,1:]

def cluster_problem(metric):
    model = SpectralClustering(1,affinity='precomputed')
    prediction = model.fit_predict(metric)
    unique, counts = np.unique(prediction, return_counts=True)

    for label, count in zip(unique, counts):
        print(f"Cluster {label}: {count} points")
    return prediction


class Local_Search():
    def __init__(self, ub_sol, ub_obj, VRP_instance):

        self.current_sol = ub_sol
        self.current_obj = ub_obj

        self.tw_scaler =VRP_instance.time_windows[0, 1]
        self.time_windows =VRP_instance.time_windows
        self.time_matrix =VRP_instance.time_matrix
        self.service_times =VRP_instance.service_times
        self.demands =VRP_instance.demands
        self.coords =VRP_instance.coords
        self.vehicle_capacity =VRP_instance.vehicle_capacity
        self.num_customers =VRP_instance.N

        print("Sol feasibility check passed")


    def optimize(self, k_opt_iter=100):
        self.insertions = {}

        current_sols = [(self.current_sol,self.current_obj)]
        start_time = time.time()
        print("Current Objective: "+str(self.current_obj))
        for k in range(k_opt_iter):
            new_sols = []
            for sol_tuple in current_sols:
                sol = sol_tuple[0]
                neighborhood = self.k_exchange(sol)
                for neighbor in neighborhood:
                    #feasible,neighbor_cost = self.check_sol_feasibility(neighbor)
                    neighbor_cost = sum([sum([self.time_matrix[route[i],route[i+1]] for i
                                         in range(len(route)-1)])for route in neighbor])
                    #if feasible:
                    new_sols.append((neighbor,neighbor_cost))
                    if neighbor_cost < self.current_obj:
                        self.current_sol = neighbor
                        self.current_obj = neighbor_cost
                        print("Current Objective: " + str(self.current_obj))
                        print("Current time: " + str(time.time() - start_time))
                        print("------------------------")

            current_sols += new_sols
            current_sols.sort(key=lambda x: x[1])
            current_sols = current_sols[:10]

        best_obj = self.current_obj
        best_sol = self.current_sol
        assert self.check_sol_feasibility(best_sol)[0]
        print("Sol feasibility check passed")
        return best_sol, best_obj

    def relatedness_removal(self,current, val_matrix,
                            nr_nodes_to_remove=None, prob=5):

        destroyed_solution = copy.deepcopy(current)
        visited_customers = [customer for route in destroyed_solution for customer in route
                             if customer !=0]

        if nr_nodes_to_remove is None:
            nr_nodes_to_remove = determine_nr_nodes_to_remove(self.num_customers)

        node_to_remove = random.choice(visited_customers)
        for route in destroyed_solution:
            while node_to_remove in route:
                route.remove(node_to_remove)
                visited_customers.remove(node_to_remove)

        for i in range(nr_nodes_to_remove - 1):
            related_nodes = []
            normalized_distances = scaled(val_matrix[node_to_remove, :])
            route_node_to_remove = [route for route in current if node_to_remove in route][0]
            for route in destroyed_solution:
                for node in route:
                    if node!=0:
                        if node in route_node_to_remove:
                            related_nodes.append((node, normalized_distances[node]))
                        else:
                            related_nodes.append((node, normalized_distances[node] + 1))

            if random.random() < 1 / prob:
                node_to_remove = random.choice(visited_customers)
            else:
                node_to_remove = min(related_nodes, key=lambda x: x[1])[0]
            for route in destroyed_solution:
                while node_to_remove in route:
                    route.remove(node_to_remove)
                    visited_customers.remove(node_to_remove)
        destroyed_solution = [route for route in destroyed_solution if (route != []
                              and route !=[0,0])]

        return destroyed_solution

    def neighbor_graph_removal(self,current, value_mat, nr_nodes_to_remove=None, prob=5):
        destroyed_solution = copy.deepcopy(current)

        if nr_nodes_to_remove is None:
            nr_nodes_to_remove = determine_nr_nodes_to_remove(self.num_customers)

        values = {}
        for route in destroyed_solution:
            if len(route) == 1:
                values[route[0]] = value_mat[0, route[0]] + value_mat[route[0], 0]
            else:
                for i in range(len(route)):
                    if i == 0:
                        values[route[i]] = value_mat[0, route[i]] + value_mat[
                            route[i], route[1]]
                    elif i == len(route) - 1:
                        values[route[i]] = value_mat[route[i - 1],
                                                                         route[i]] + value_mat[
                            route[i], 0]
                    else:
                        values[route[i]] = value_mat[route[i - 1],
                                                                         route[i]] + value_mat[
                            route[i], route[i + 1]]

        removed_nodes = []
        while len(removed_nodes) < nr_nodes_to_remove:
            try:
                del values[0]
            except:
                print(destroyed_solution)
                sys.exit(0)
            # sort the nodes based on their neighbor graph scores in descending order
            sorted_nodes = sorted(values.items(), key=lambda x: x[1], reverse=True)
            # select the node to remove
            removal_option = 0
            while random.random() < 1 / prob and removal_option < len(sorted_nodes) - 1:
                removal_option += 1
            node_to_remove, score = sorted_nodes[removal_option]


            # remove the node from its route
            for route in destroyed_solution:
                if node_to_remove in route:
                    route.remove(node_to_remove)
                    removed_nodes.append(node_to_remove)
                    if route == [0, 0]:
                        destroyed_solution.remove(route)

                    values.pop(node_to_remove)
                    if len(route) == 0:
                        destroyed_solution.remove([])

                    elif len(route) == 1:
                        values[route[0]] = value_mat[0, route[0]] + value_mat[
                            route[0], 0]
                    else:
                        for i in range(len(route)):
                            if i == 0:
                                values[route[i]] = value_mat[0, route[
                                    i]] + value_mat[route[i], route[1]]
                            elif i == len(route) - 1:
                                values[route[i]] = value_mat[route[i - 1], route[
                                    i]] + value_mat[route[i], 0]
                            else:
                                values[route[i]] = value_mat[route[i - 1], route[
                                    i]] + value_mat[route[i], route[i + 1]]
                    break
        return destroyed_solution

    def get_regret_single_insertion(self, routes, customer,val_matrix):
        relevant_insertions = {}
        for route_idx in range(len(routes)):
            for i in range(len(routes[route_idx])+1):
                dict_key = (customer, i, tuple(routes[route_idx]))
                if dict_key not in self.insertions:
                    updated_route = routes[route_idx][:i] + [customer] + routes[route_idx][i:]
                    if check_route_feasibility(updated_route, self.time_matrix, self.time_windows,
                                            self.service_times, self.demands,
                                            self.vehicle_capacity):
                        if i == 0:
                            cost_difference = val_matrix[0, updated_route[0]] + val_matrix[
                                updated_route[0], updated_route[1]] - val_matrix[0, updated_route[1]]
                        elif i == len(routes[route_idx]):
                            cost_difference = val_matrix[updated_route[-1], 0] + val_matrix[
                                updated_route[i - 1], updated_route[i]] - val_matrix[updated_route[i - 1], 0]
                        else:
                            cost_difference = val_matrix[updated_route[i - 1], updated_route[i]] + \
                                              val_matrix[updated_route[i], updated_route[i + 1]] - \
                                              val_matrix[updated_route[i - 1], updated_route[i + 1]]

                        self.insertions[dict_key] = cost_difference
                        relevant_insertions[dict_key] = cost_difference

                    else:
                        self.insertions[dict_key] = False
                        relevant_insertions[dict_key] = False

                else:
                    relevant_insertions[dict_key] = self.insertions[dict_key]

        relevant_insertions = {key: relevant_insertions[key] for key in relevant_insertions if
                               relevant_insertions[key] != False}

        if len(relevant_insertions) == 1:
            best_insertion = min(relevant_insertions, key=relevant_insertions.get)
            return best_insertion, 0

        elif len(relevant_insertions) > 1:
            best_insertion = min(relevant_insertions, key=relevant_insertions.get)

            if len(set(relevant_insertions.values())) == 1:  # when all options are of equal value:
                regret = 0
            else:
                regret = sorted(list(relevant_insertions.values()))[1] - min(relevant_insertions.values())
            return best_insertion, regret
        else:
            # no insertions possible for this customer
            return -1, -1

    def regret_insertion(self,current, val_matrix, prob=1.5):
        visited_customers = [customer for route in current for customer in route]
        all_customers = set(range(1, self.num_customers + 1))
        unvisited_customers = all_customers - set(visited_customers)

        repaired = copy.deepcopy(current)
        while unvisited_customers:
            insertion_options = {}
            for customer in unvisited_customers:
                best_insertion, regret = self.get_regret_single_insertion(repaired, customer,
                                                                          val_matrix)
                if best_insertion != -1:
                    insertion_options[best_insertion] = regret

            if not insertion_options:
                repaired.append([0,random.choice(list(unvisited_customers)),0])
            else:
                insertion_option = 0
                while random.random() < 1 / prob and insertion_option < len(insertion_options) - 1:
                    insertion_option += 1

                insertion_operation = sorted(insertion_options, reverse=True)[insertion_option]
                customer = insertion_operation[0]
                customer_index = insertion_operation[1]
                route = list(insertion_operation[2])
                route_index = repaired.index(route)
                repaired[route_index].insert(customer_index, customer)

            visited_customers = [customer for route in repaired for customer in route]
            unvisited_customers = all_customers - set(visited_customers)
        return repaired

    def k_exchange(self, current_sol, num_neighbors =1):
        neighborhood = []

        for x in range(num_neighbors):
            destroyed_sol = self.relatedness_removal(current_sol,self.time_matrix)
            new_sol = self.regret_insertion(destroyed_sol,self.time_matrix)
            neighborhood.append(new_sol)

        return neighborhood

    def check_sol_feasibility(self, sol):
        cust_covered = []
        obj = 0

        for route in sol:
            if not check_route_feasibility(route, self.time_matrix, self.time_windows,
                                       self.service_times, self.demands,
                                       self.vehicle_capacity):
                print(route)
                return False, math.inf
            obj += sum([self.time_matrix[route[x],route[x+1]] for x in range(len(route)-1)])
            cust_covered +=route[1:-1]

        cust_covered = list(set(cust_covered))
        assert len(cust_covered)==self.num_customers
        cust_covered.sort()
        assert cust_covered[0] == 1
        assert cust_covered[-1]== self.num_customers
        return True, obj



def construct_cvrptw_sol(VRP_instance, model_params, model_load,int_sol=False):
    forbidden_edges = []
    compelled_edges = []

    num_customers = VRP_instance.N
    vehicle_capacity = VRP_instance.vehicle_capacity
    time_matrix = VRP_instance.time_matrix
    service_times = VRP_instance.service_times
    time_windows = VRP_instance.time_windows
    demands = VRP_instance.demands

    initial_routes, initial_costs, initial_orders = initialize_columns(num_customers, vehicle_capacity, time_matrix,
                                                                       service_times, time_windows,
                                                                       demands)
    total_init_cost = sum(cost for cost in initial_costs)
    print("The initial total cost: "+str(total_init_cost))
    ub_sol = initial_orders
    ub_obj =total_init_cost

    cg_with_rl = CG_with_RL(VRP_instance,model_params, model_load,
                            initial_routes, initial_costs,initial_orders)
    lb_sol, lb_obj, routes, costs, orders, _ =cg_with_rl.solve(forbidden_edges, compelled_edges)

    if int_sol:
        ub_sol, ub_obj, _ = generate_upper_bound_2(lb_sol, costs,
                                                num_customers,time_matrix, routes)
        if ub_obj > total_init_cost:
            ub_obj = total_init_cost
            ub_sol = initial_orders

    edge_scores = determine_branching_rule(lb_sol,False)
    print("Initial integer solution has objective: "+str(ub_obj))
    return ub_sol,ub_obj, edge_scores


def main():
    random.seed(10)
    np.random.seed(10)

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_customers', type=int, default=200)
    example_path = 'C:/Users/abdug/PycharmProjects/POMO-implementation/ESPRCTW/POMO/result/100_new_instances'
    parser.add_argument('--RL_model_path', type=str, default=example_path)
    parser.add_argument('--k_opt_iter', type=int, default=20)
    args = parser.parse_args()
    num_customers = args.num_customers
    RL_path = args.RL_model_path
    k_opt_iter = args.k_opt_iter

    RL_model_params = {
        'embedding_dim': 128,
        'sqrt_embedding_dim': 128 ** (1 / 2),
        'encoder_layer_num': 6,
        'qkv_dim': 16,
        'head_num': 8,
        'logit_clipping': 10,
        'ff_hidden_dim': 512,
        'eval_type': 'argmax',
    }

    RL_model_load = {
        'path': RL_path,
        'epoch': 200}

    file = "config.json"
    with open(file, 'r') as f:
        config = json.load(f)

    print("Instances of size: " + str(num_customers))
    results = []
    for experiment in range(1):
        VRP_instance = Instance_Generator(N=num_customers)
        time_1 = time.time()
        sol,obj,edge_scores = construct_cvrptw_sol(VRP_instance, RL_model_params, RL_model_load,
                                                    int_sol = True)
        print("CG run time: " + str(time.time() - time_1))

        arc_dist = determine_arc_distribution(edge_scores,num_customers)
        cus_region = cluster_problem(arc_dist)
        subproblems = partion_problem(VRP_instance,cus_region)

        total_obj = 0
        for problem in subproblems:
            #sol, obj, _ = construct_cvrptw_sol(problem, RL_model_params, RL_model_load, int_sol = True)
            ls = Local_Search(sol,obj, problem)
            best_sol, best_obj = ls.optimize(k_opt_iter=k_opt_iter)
            total_obj+=best_obj

        print("Final objective for instance "+str(experiment)+": " + str(total_obj)+'\n')
        print("with run time: "+str(time.time()-time_1))

        results.append(total_obj)

    mean_obj = statistics.mean(results)
    std_obj = statistics.stdev(results)
    print("The mean objective value is: " + str(mean_obj))
    print("The std dev. objective is: " + str(std_obj))

    pickle_out = open('Exact Results N=' + str(num_customers), 'wb')
    pickle.dump(results, pickle_out)
    pickle_out.close()



if __name__ == "__main__":
    main()