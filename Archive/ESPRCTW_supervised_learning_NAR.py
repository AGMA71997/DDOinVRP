from sklearn.neural_network import MLPClassifier as NN
import numpy
import random
from instance_generator import Instance_Generator
from column_generation import initialize_columns, MasterProblem, Subproblem, convert_ordered_route
import sys


def build_training_data(number_of_training_instances, num_customers):

    print("Training Started")
    for i in range(number_of_training_instances):
        iter = 0
        X_sub = {}
        Y_sub = {}

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
        added_orders = initial_orders.copy()
        while True:
            master_problem.solve()
            duals = master_problem.retain_duals()
            subproblem = Subproblem(num_customers, vehicle_capacity, time_matrix, demands, time_windows, time_limit,
                                    duals, service_times, forbidden_edges, compelled_edges)
            ordered_route, reduced_cost = subproblem.solve()
            print("RC is " + str(reduced_cost))
            print(ordered_route)
            demands_reshape = numpy.reshape(demands, (len(demands), 1))
            service_times_reshape = numpy.reshape(demands, (len(service_times), 1))
            X_sub[iter] = numpy.concatenate((subproblem.price, time_windows, demands_reshape,
                                             service_times_reshape),axis=1).flatten()
            edge_selection=edge_matrix(ordered_route, num_customers)
            Y_sub[iter] = edge_selection.flatten()
            cost = sum(time_matrix[ordered_route[i], ordered_route[i + 1]] for i in range(len(ordered_route) - 1))
            route = convert_ordered_route(ordered_route, num_customers)
            if reduced_cost < 0 and ordered_route not in added_orders:
                master_problem.add_columns([route], [cost], [ordered_route], forbidden_edges, compelled_edges)
                added_orders.append(ordered_route)
                iter += 1
            else:
                print("Addition Failed")
                break

        try:
            X = numpy.concatenate((X, numpy.array([X_sub[i] for i in range(iter)])), axis=0)
            Y = numpy.concatenate((Y, numpy.array([Y_sub[i] for i in range(iter)])), axis=0)
        except:
            X=numpy.array([X_sub[i] for i in range(iter)])
            Y=numpy.array([Y_sub[i] for i in range(iter)])

    return X, Y


def train_ML_model(X, Y):
    neural_network = NN(max_iter=1000)
    neural_network.fit(X, Y)
    return neural_network


def edge_matrix(ordered_route, num_customers):
    edges = numpy.zeros((num_customers+1, num_customers+1))
    for i in range(len(ordered_route) - 1):
        edges[ordered_route[i], ordered_route[i + 1]] = 1
    return edges

def retain_route_from_probabilities(probabilities,time_windows,demands,service_times,travel_times,
                                    time_limit,vehicle_capacity,num_customers):
    probas=numpy.copy(probabilities)
    probas=numpy.reshape(probas,(num_customers+1,(num_customers+1)))
    route=[0]
    last_visited=0
    remaining_time=time_limit
    remaining_capcity=vehicle_capacity
    current_time=0
    visited_customers=[]
    while True:
        possible_addition=False
        while not possible_addition:
            next_visit=numpy.argmax(probas[last_visited,:])
            proposed_arrival_time=current_time+travel_times[last_visited,next_visit]
            if proposed_arrival_time>time_windows[next_visit][1] or remaining_capcity<demands[next_visit] or \
                    remaining_time<travel_times[last_visited,next_visit]+service_times[next_visit]+\
                    travel_times[next_visit,0] or next_visit in visited_customers:
                probas[last_visited,next_visit]=0
                continue
            else:
                possible_addition=True
                route.append(next_visit)
                current_time=max(current_time,time_windows[next_visit,0])
                remaining_time-=travel_times[last_visited,next_visit]+service_times[next_visit]
                remaining_capcity-=demands[next_visit]
                last_visited=next_visit
        if last_visited!=0:
            visited_customers.append(last_visited)
        else:
            return route

def main():
    random.seed(5)
    numpy.random.seed(25)

    num_customers = 10
    X, Y = build_training_data(50, num_customers)
    model = train_ML_model(X, Y)

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

    iter=0
    while True:
        master_problem.solve()
        duals = master_problem.retain_duals()
        subproblem = Subproblem(num_customers, vehicle_capacity, time_matrix, demands, time_windows, time_limit,
                                duals, service_times, forbidden_edges, compelled_edges)
        ordered_route, reduced_cost = subproblem.solve()
        demands_reshape = numpy.reshape(demands, (len(demands), 1))
        service_times_reshape = numpy.reshape(demands, (len(service_times), 1))
        feature_vector = numpy.concatenate(
            (subproblem.price, time_windows, demands_reshape, service_times_reshape), axis=1).flatten()
        feature_vector=numpy.reshape(feature_vector,(1,len(feature_vector)))
        probabilities = model.predict_proba(feature_vector)
        predicted_route=retain_route_from_probabilities(probabilities,time_windows,demands,service_times,
                                                        time_matrix,time_limit,vehicle_capacity,num_customers)
        predicted_reduced_cost=sum(subproblem.price[predicted_route[i],predicted_route[i+1]] for i in range(len(predicted_route)-1))
        print("The predicted route is "+str(predicted_route))
        print("with reduced cost "+str(predicted_reduced_cost))
        print(numpy.reshape(probabilities,(num_customers+1,num_customers+1)))
        print("The optimal route is " + str(ordered_route))
        print("with reduced cost " + str(reduced_cost))
        print("--------------")
        cost = sum(time_matrix[predicted_route[i], predicted_route[i + 1]] for i in range(len(predicted_route) - 1))
        route = convert_ordered_route(predicted_route, num_customers)
        if predicted_reduced_cost < 0 and iter<5:
            master_problem.add_columns([route], [cost], [predicted_route], forbidden_edges, compelled_edges)
            iter += 1
        else:
            print("Experiment over")
            break


if __name__ == "__main__":
    main()