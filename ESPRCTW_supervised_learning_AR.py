from sklearn.neural_network import MLPClassifier as NN
from sklearn.preprocessing import StandardScaler

import numpy
import random
from instance_generator import Instance_Generator
from column_generation import initialize_columns, MasterProblem, convert_ordered_route
import sys
import math


class Data_Generator(object):

    def __init__(self):
        pass

    def generate_dataset(self, number_of_training_instances, num_customers):

        for i in range(number_of_training_instances):
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

            if initial_routes == []:
                initial_routes, initial_costs, initial_orders = initialize_columns(num_customers, time_matrix)
            master_problem = MasterProblem(num_customers, initial_routes, initial_costs, initial_orders,
                                           forbidden_edges,
                                           compelled_edges)
            added_orders = initial_orders.copy()
            while True:
                master_problem.solve()
                duals = master_problem.retain_duals()
                subproblem = Subproblem_ML_Varaint(num_customers, vehicle_capacity, time_matrix, demands, time_windows,
                                                   time_limit,
                                                   duals, service_times, forbidden_edges, compelled_edges)
                ordered_route, reduced_cost, transitions = subproblem.solve()
                X_sub, Y_sub = self.process_transitions(transitions)
                print("RC is " + str(reduced_cost))
                print(ordered_route)
                cost = sum(time_matrix[ordered_route[i], ordered_route[i + 1]] for i in range(len(ordered_route) - 1))
                route = convert_ordered_route(ordered_route, num_customers)
                if reduced_cost < 0 and ordered_route not in added_orders:
                    master_problem.add_columns([route], [cost], [ordered_route], forbidden_edges, compelled_edges)
                    added_orders.append(ordered_route)
                else:
                    print("Addition Failed")
                    break
            try:
                X = numpy.concatenate((X, X_sub), axis=0)
                Y = numpy.concatenate((Y, Y_sub), axis=0)
            except:
                X = X_sub
                Y = Y_sub

        return X, Y

    def process_transitions(self, transitions):

        indx = 0
        for key in transitions:
            features = transitions[key]
            added_nodes, RT, processing_times, RC, demands, time_left_in_window, prices = features
            nodes_in = numpy.zeros(len(demands))
            nodes_in[added_nodes] = 1
            time_consumption = processing_times
            capacity_consumption =  demands
            time_left_in_window = time_left_in_window < 0
            time_left_in_window = time_left_in_window.astype(int)
            time_left_in_window = time_left_in_window[:, 0] + time_left_in_window[:, 1]
            window_allows = time_left_in_window < 2
            window_allows = window_allows.astype(int)
            decision = numpy.zeros(len(demands))
            decision[key] = 1
            feature_vector = numpy.concatenate(
                (nodes_in, time_consumption, capacity_consumption, window_allows, prices), axis=0)
            if indx == 0:
                N = len(transitions)
                X = numpy.zeros((N, len(feature_vector)))
                Y = numpy.zeros((N, len(decision)))
            X[indx, :] = feature_vector
            Y[indx, :] = decision
            indx += 1

        return X, Y


class Subproblem_ML_Varaint(object):
    def __init__(self, num_customers, vehicle_capacity, time_matrix, demands, time_windows, time_limit, duals,
                 service_times, forbidden_edges, compelled_edges):
        self.num_customers = num_customers
        self.vehicle_capacity = vehicle_capacity
        self.time_matrix = time_matrix
        self.demands = demands
        self.time_windows = time_windows
        self.time_limit = time_limit
        self.duals = duals
        self.service_times = service_times
        self.price = numpy.zeros((num_customers + 1, num_customers + 1))

        for i in range(num_customers + 1):
            for j in range(num_customers + 1):
                if i != j:
                    if i != 0:
                        self.price[i, j] = time_matrix[i, j] - duals[i - 1]
                    else:
                        self.price[i, j] = time_matrix[i, j]

                    edge = [i, j]
                    if edge in forbidden_edges:
                        self.price[i, j] = math.inf
                    elif edge in compelled_edges:
                        self.price[i, j] = -100000

    def dynamic_program(self, start_point, current_label, unvisited_customers, remaining_time, remaining_capacity,
                        current_time, current_price, current_X):

        if current_time > self.time_windows[start_point, 1]:
            return [], math.inf, {}
        if start_point != 0:
            if current_time < self.time_windows[start_point, 0]:
                current_time = self.time_windows[start_point, 0]

        current_time += self.service_times[start_point]
        remaining_time -= self.service_times[start_point]

        if remaining_time < 0 or remaining_capacity < 0:
            return [], math.inf, {}

        if current_label[0] == 0 and current_label[-1] == 0 and len(current_label) > 1:
            return current_label, current_price, current_X

        best_label = []
        best_price = math.inf
        best_transitions = {}
        for j in unvisited_customers:
            if j != start_point:
                copy_label = current_label.copy()
                copy_unvisited = unvisited_customers.copy()
                X_copy = current_X.copy()
                RT = remaining_time
                RC = remaining_capacity
                CT = current_time
                CP = current_price

                X_copy[j] = [copy_label[1:], RT, self.time_matrix[start_point, :] + self.service_times, RC,
                             self.demands, self.time_windows - CT, self.price[start_point, :]]
                copy_label.append(j)
                copy_unvisited.remove(j)
                RT -= self.time_matrix[start_point, j]
                RC -= self.demands[j]
                CT += self.time_matrix[start_point, j]
                CP += self.price[start_point, j]

                label, price, X = self.dynamic_program(j, copy_label, copy_unvisited, RT, RC, CT, CP, X_copy)
                if price < best_price:
                    best_price = price
                    best_label = label
                    best_transitions = X

        return best_label, best_price, best_transitions

    def autoregressive_ml_addition(self, start_point, current_label, remaining_time,
                                   remaining_capacity, current_time, current_price, ML_model):

        if current_label[0] == 0 and current_label[-1] == 0 and len(current_label) > 1:
            return current_label, current_price

        copy_label = current_label.copy()

        transition = {}
        transition[0] = [copy_label[1:], remaining_time, self.time_matrix[start_point, :] + self.service_times, remaining_capacity,
                         self.demands, self.time_windows - current_time, self.price[start_point, :]]
        X, DummY = Data_Generator.process_transitions(self,transition)
        probas=ML_model.predict_proba(X)

        feasible=False
        while not feasible:
            j  = numpy.argmax(probas[0])
            feasible= self.check_addition_feasibility(copy_label,j,remaining_capacity,remaining_time,current_time)
            if not feasible:
                probas[0,j]=0

        copy_label.append(j)
        process_time=self.time_matrix[start_point,j]+self.service_times[j]
        current_time=max(current_time,self.time_windows[j,0])
        remaining_time -= process_time
        remaining_capacity -= self.demands[j]
        current_time += process_time
        current_price += self.price[start_point, j]

        label, price = self.autoregressive_ml_addition(j, copy_label, remaining_time, remaining_capacity,
                                                       current_time, current_price, ML_model)

        return label, price

    def check_addition_feasibility(self,label,node,RC,RT,CT):
        i = label[-1]
        total_return_time = self.time_matrix[i, node] + self.service_times[node] + self.time_matrix[node, 0]
        if (CT+self.time_matrix[i,node]>self.time_windows[node,1] or RC<self.demands[node] or RT<total_return_time \
                or node in label or CT+total_return_time>self.time_windows[0,1]) and node!=0:
            return False
        print(str(node)+" added")
        return True

    def solve(self):
        start_point = 0
        current_label = [0]
        unvisited_customers = list(range(0, self.num_customers + 1))
        remaining_time = self.time_limit
        remaining_capacity = self.vehicle_capacity
        current_time = 0
        current_price = 0
        X = {}
        best_route, best_price, best_transitions = self.dynamic_program(start_point, current_label, unvisited_customers,
                                                                        remaining_time,
                                                                        remaining_capacity, current_time, current_price,
                                                                        X)

        return best_route, best_price, best_transitions

    def solve_with_ML(self, ML_model):
        start_point = 0
        current_label = [0]
        remaining_time = self.time_limit
        remaining_capacity = self.vehicle_capacity
        current_time = 0
        current_price = 0
        label, price = self.autoregressive_ml_addition(start_point, current_label, remaining_time,
                                                       remaining_capacity, current_time, current_price, ML_model)
        return label, price

class ML_solver_ESPRCTW(object):

    def __init__(self):
        self.neural_network = NN(max_iter=1000)

    def train_ML_model(self, X, Y):
        self.neural_network.fit(X, Y)

    def retain_route_from_probabilities(self, probabilities, time_windows, demands, service_times, travel_times,
                                        time_limit, vehicle_capacity, num_customers):
        probas = numpy.copy(probabilities)
        probas = numpy.reshape(probas, (num_customers + 1, (num_customers + 1)))
        route = [0]
        last_visited = 0
        remaining_time = time_limit
        remaining_capcity = vehicle_capacity
        current_time = 0
        visited_customers = []
        while True:
            possible_addition = False
            while not possible_addition:
                next_visit = numpy.argmax(probas[last_visited, :])
                proposed_arrival_time = current_time + travel_times[last_visited, next_visit]
                if proposed_arrival_time > time_windows[next_visit][1] or remaining_capcity < demands[next_visit] or \
                        remaining_time < travel_times[last_visited, next_visit] + service_times[next_visit] + \
                        travel_times[next_visit, 0] or next_visit in visited_customers:
                    probas[last_visited, next_visit] = 0
                    continue
                else:
                    possible_addition = True
                    route.append(next_visit)
                    current_time = max(current_time, time_windows[next_visit, 0])
                    remaining_time -= travel_times[last_visited, next_visit] + service_times[next_visit]
                    remaining_capcity -= demands[next_visit]
                    last_visited = next_visit
            if last_visited != 0:
                visited_customers.append(last_visited)
            else:
                return route

    def test_model(self):
        print("Started Testing")
        num_customers = 10

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

        if initial_routes == []:
            initial_routes, initial_costs, initial_orders = initialize_columns(num_customers, time_matrix)
        master_problem = MasterProblem(num_customers, initial_routes, initial_costs, initial_orders, forbidden_edges,
                                       compelled_edges)

        iter = 0
        while True:
            master_problem.solve()
            duals = master_problem.retain_duals()
            subproblem = Subproblem_ML_Varaint(num_customers, vehicle_capacity, time_matrix, demands, time_windows,
                                               time_limit,
                                               duals, service_times, forbidden_edges, compelled_edges)
            predicted_route, predicted_reduced_cost = subproblem.solve_with_ML(self.neural_network)
            predicted_reduced_cost = sum(
                subproblem.price[predicted_route[i], predicted_route[i + 1]] for i in range(len(predicted_route) - 1))
            print("The predicted route is " + str(predicted_route))
            print("with reduced cost " + str(predicted_reduced_cost))
            ordered_route, reduced_cost, transitions = subproblem.solve()
            print("The optimal route is " + str(ordered_route))
            print("with reduced cost " + str(reduced_cost))
            print("--------------")
            cost = sum(time_matrix[predicted_route[i], predicted_route[i + 1]] for i in range(len(predicted_route) - 1))
            route = convert_ordered_route(predicted_route, num_customers)
            if predicted_reduced_cost < 0 and iter < 5:
                master_problem.add_columns([route], [cost], [predicted_route], forbidden_edges, compelled_edges)
                iter += 1
            else:
                print("Experiment over")
                break


def main():
    random.seed(5)
    numpy.random.seed(25)
    number_of_instances = 50
    num_customers = 10

    DG = Data_Generator()
    X, Y = DG.generate_dataset(number_of_instances, num_customers)
    ml_model = ML_solver_ESPRCTW()
    ml_model.train_ML_model(X, Y)
    ml_model.test_model()


if __name__ == "__main__":
    main()
