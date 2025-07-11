from instance_generator import Instance_Generator
import time
from utils import *

import json
import random
import numpy as np
import statistics
import argparse
import gurobipy as gp
from gurobipy import GRB
from itertools import combinations


def solve_exact(VRP_instance):
    travel_times = VRP_instance.time_matrix

    time_windows = VRP_instance.time_windows
    demands = VRP_instance.demands
    vehicle_capacity = VRP_instance.vehicle_capacity
    service_times = VRP_instance.service_times
    num_customers = VRP_instance.N

    nodes = range(num_customers + 1)
    print({i: demands[i] for i in nodes})
    customers = range(1, num_customers + 1)

    model = gp.Model("CVRPTW_LP")

    # Variables
    x = model.addVars(nodes, nodes, vtype=GRB.BINARY, lb=0, ub=1, name="x")
    t = model.addVars(nodes, vtype=GRB.CONTINUOUS, name="t", lb=0)  # Arrival time
    u = model.addVars(nodes, vtype=GRB.CONTINUOUS, name="u", lb=0)  # Load upon arrival

    # Objective: minimize total travel time
    model.setObjective(gp.quicksum(travel_times[i][j] * x[i, j]
                                   for i in nodes for j in nodes if i != j), GRB.MINIMIZE)

    for j in customers:  # Must go somewhere
        model.addConstr(gp.quicksum(x[i, j] for i in nodes if i != j) == 1)

    # No loops
    for i in nodes:
        model.addConstr(x[i, i] == 0)

    # Flow conservation
    for i in nodes:
        if i == 0:
            model.addConstr(gp.quicksum(x[i, j] for j in nodes if j != i) >= 1)
        else:
            model.addConstr(gp.quicksum(x[i, j] for j in nodes if j != i) ==
                            gp.quicksum(x[j, i] for j in nodes if j != i))

    big_M = max([tw[1] for tw in time_windows]) + max(service_times) + max([max(row) for row in travel_times])
    for i in nodes:  # Time window constraints
        model.addConstr(t[i] >= time_windows[i][0])
        model.addConstr(t[i] <= time_windows[i][1])
        for j in nodes:  # Time propagation constraints
            if i != j:
                if i == 0:
                    model.addConstr(t[j] >= travel_times[i][j]*x[i, j] - big_M * (1 - x[i, j]))
                else:
                    model.addConstr(t[j] >= t[i] + (service_times[i] + travel_times[i][j])*x[i, j] - big_M * (1 - x[i, j]))

    for i in nodes:  # Load initialization
        if i == 0:
            model.addConstr(u[0] == 0)
        else:  # Load bounds
            model.addConstr(u[i] <= vehicle_capacity)
            model.addConstr(u[i] >= demands[i])

        for j in customers:  # Capacity constraints (cumulative load propagation)
            if i != j:
                model.addConstr(u[j] >= u[i] + demands[j] * (x[i, j]) - vehicle_capacity * (1 - x[i, j]))

    # Solve LP
    # model.setParam('OutputFlag', 0)
    model.update()
    model.optimize()

    x_sol = {(i, j): round(x[i, j].X, 2) for i in nodes for j in nodes if i != j and x[i, j].X > 0.0001}
    u_sol = {i: u[i].x for i in nodes}
    '''print(x_sol)
    print(u_sol)
    print(model.objval)
    subtourelim(model, x_sol, x)'''

    # Extract results
    counter = 0
    while True:
        x_sol = {(i, j): round(x[i, j].X, 2) for i in nodes for j in nodes if i != j and x[i, j].X > 0.0001}
        if subtourelim(model, x_sol, x):
            model.optimize()
            counter += 1
        else:
            break
        if counter % 10 == 0:
            # print(x_sol)
            print("Round: " + str(counter))
            print(model.objval)
            print("----------------")

    return x_sol, model.ObjVal


def subtourelim(model, solution, X):
    # make a list of edges selected in the solution
    unvisited = list(set([edge[0] for edge in solution] + [edge[1] for edge in solution]))
    unvisited.append(0)
    # find the shortest cycle in the selected edge list
    tour = subtour(solution, unvisited)
    if len(tour) > 0:
        # add subtour elimination constr. for every pair of cities in tour
        model.addConstr(gp.quicksum(X[i, j] for (i, j) in solution if j in tour) <= 0.95)
        model.update()
        # model.reset()
        return True
    return False


# Given a tuplelist of edges, find the shortest subtour
def subtour(edges, unvisited):
    cycle = range(len(unvisited) + 1)  # initial length has 1 more city
    while unvisited:  # true if list is non-empty
        thiscycle = []
        neighbors = unvisited
        while neighbors:
            current = neighbors[0]
            thiscycle.append(current)
            unvisited.remove(current)
            neighbors = [j for i, j in edges if i == current and j in unvisited]
        if len(cycle) > len(thiscycle):
            cycle = thiscycle
    print(cycle)
    return cycle


def main():
    random.seed(10)
    np.random.seed(10)

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_customers', type=int, default=100)
    args = parser.parse_args()
    num_customers = args.num_customers

    file = "config.json"
    with open(file, 'r') as f:
        config = json.load(f)

    print("Instances of size: " + str(num_customers))
    results = []
    for experiment in range(1):
        VRP_instance = Instance_Generator(N=num_customers)
        orders, obj = solve_exact(VRP_instance)

        print("objective: " + str(obj))
        print("number of columns: " + str(len(orders)))

        results.append(obj)

    mean_obj = statistics.mean(results)
    std_obj = statistics.stdev(results)
    print("The mean objective value is: " + str(mean_obj))
    print("The std dev. objective is: " + str(std_obj))

    pickle_out = open('Exact Results N=' + str(num_customers), 'wb')
    pickle.dump(results, pickle_out)
    pickle_out.close()


if __name__ == "__main__":
    main()
