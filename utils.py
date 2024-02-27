import pickle
import os
import numpy
import torch
import json
import math
import matplotlib.pyplot as pp


def create_price(time_matrix, duals):
    if len(duals) < len(time_matrix):
        duals.insert(0, 0)
    assert duals[0] == 0 and len(duals) == len(time_matrix)
    duals = numpy.array(duals)
    prices = (time_matrix - duals) * -1
    numpy.fill_diagonal(prices, math.inf)
    return prices


def reshape_problem(coords, demands, time_windows, duals, service_times, time_matrix, prices):
    coords = numpy.copy(coords)
    demands = numpy.copy(demands)
    time_windows = numpy.copy(time_windows)
    duals = duals.copy()
    service_times = numpy.copy(service_times)
    time_matrix = numpy.copy(time_matrix)
    prices = numpy.copy(prices)

    remaining_customers = []
    for x in range(1, len(coords)):
        if coords[x, 0] == math.inf:
            demands[x] = math.inf
            time_windows[x, :] = math.inf
            duals[x] = math.inf
            service_times[x] = math.inf
            time_matrix[x, :] = math.inf
            time_matrix[:, x] = math.inf
            prices[x, :] = math.inf
            prices[:, x] = math.inf
        else:
            remaining_customers.append(x)

    cus_mapping = {}
    for x in range(len(remaining_customers)):
        cus_mapping[x + 1] = remaining_customers[x]

    coords = coords[coords[:, 0] != math.inf]
    demands = demands[demands[:] != math.inf]
    time_windows = time_windows[time_windows[:, 0] != math.inf]
    duals = [duals[x] for x in range(len(duals)) if duals[x] != math.inf]
    service_times = service_times[service_times[:] != math.inf]
    time_matrix = time_matrix[time_matrix[:, 0] != math.inf]
    mask = (time_matrix == math.inf)
    idx = mask.any(axis=0)
    time_matrix = time_matrix[:, ~idx]
    prices = prices[prices[:, 0] != math.inf]
    mask = (prices == math.inf)
    idx = mask.any(axis=0)
    prices = prices[:, ~idx]

    # print("The problem has been reduced to size: " + str(len(coords) - 1))
    return coords, demands, time_windows, duals, service_times, time_matrix, prices, cus_mapping


def remap_route(route, cus_mapping):
    for x in range(1, len(route) - 1):
        route[x] = cus_mapping[route[x]]
    return route


def data_iterator(config, POMO, num_customers, heuristic, solomon):
    if heuristic:
        directory = config["storge_directory_raw_heuristic"] + "/" + str(num_customers)
    else:
        directory = config["storge_directory_raw"] + "/" + str(num_customers)
    CL, TML, TWL, DL, STL, VCL, DUL = [], [], [], [], [], [], []
    data_count = 0
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        pickle_in = open(f, 'rb')
        cl, tml, twl, dl, stl, vcl, dul = pickle.load(pickle_in)
        data_count += len(cl)
        CL += cl
        TML += tml
        TWL += twl
        DL += dl
        STL += stl
        VCL += vcl
        DUL += dul
    print(data_count)

    if POMO:
        process_data_for_POMO(CL, TML, TWL, DL, STL, VCL, DUL, num_customers, config, heuristic, solomon)
    '''else:
        os.chdir(config['SB3 Data'])
        pickle_out = open('ESPRCTW_Data_' + str(num_customers), 'wb')
        pickle.dump([TML, TWL, DL, STL, VCL, DUL], pickle_out)
        pickle_out.close()'''


def process_data_for_POMO(CL, TML, TWL, DL, STL, VCL, DUL, num_customers, config, heuristic, solomon):
    depot_CL = []
    depot_TW = []
    PL = []
    max_dual = 0
    cl_scaler = 1
    for x in range(len(CL)):
        depot_CL.append(CL[x][0, :])
        tw_scaler = TWL[x][0, 1]
        depot_TW.append(TWL[x][0, :] / tw_scaler)
        PL.append(create_price(TML[x], DUL[x]))

        if solomon:
            cl_scaler = 100
        CL[x] = numpy.delete(CL[x], 0, 0) / cl_scaler
        TWL[x] = numpy.delete(TWL[x], 0, 0) / tw_scaler
        TML[x] = TML[x] / tw_scaler
        DL[x] = numpy.delete(DL[x], 0, 0) / VCL[x]
        STL[x] = numpy.delete(STL[x], 0, 0) / tw_scaler
        if max(DUL[x]) > max_dual:
            max_dual = max(DUL[x])
        DUL[x] = numpy.delete(DUL[x], 0, 0)
        min_val = numpy.min(PL[x])
        max_val = numpy.max(PL[x])
        PL[x] = PL[x] / max(abs(max_val), abs(min_val))

    print(max_dual)
    depot_CL = torch.tensor(numpy.stack(depot_CL), dtype=torch.float32)
    depot_TW = torch.tensor(numpy.stack(depot_TW), dtype=torch.float32)
    CL = torch.tensor(numpy.stack(CL), dtype=torch.float32)
    TWL = torch.tensor(numpy.stack(TWL), dtype=torch.float32)
    TML = torch.tensor(numpy.stack(TML), dtype=torch.float32)
    DL = torch.tensor(numpy.stack(DL), dtype=torch.float32)
    STL = torch.tensor(numpy.stack(STL), dtype=torch.float32)
    DUL = torch.tensor(numpy.stack(DUL) / max_dual, dtype=torch.float32)
    PL = torch.tensor(numpy.stack(PL), dtype=torch.float32)

    depot_CL = depot_CL[:, None, :].expand(-1, 1, -1)
    depot_TW = depot_TW[:, None, :].expand(-1, 1, -1)
    dict = {'depot_xy': depot_CL,
            'node_xy': CL,
            'node_demand': DL,
            'time_windows': TWL,
            'depot_time_window': depot_TW,
            'duals': DUL,
            'service_times': STL,
            'travel_times': TML,
            'prices': PL}

    if heuristic:
        os.chdir(config["POMO Data Heuristic"] + "/" + str(num_customers))
    else:
        os.chdir(config["POMO Data"] + "/" + str(num_customers))
    torch.save(dict, 'ESPRCTW_Data_' + str(num_customers))


def initialize_columns(num_customers, truck_capacity, time_matrix, service_times, time_windows, demands):
    unvisited_customers = list(range(1, num_customers + 1))
    solution = []
    current_stop = 0
    current_route = [0]
    remaining_capacity = truck_capacity
    current_time = 0
    while len(unvisited_customers) > 0:
        nearest_customers = numpy.argsort(time_matrix[current_stop, :].copy())
        i = 0
        feasible_addition = False

        while not feasible_addition:
            new_stop = nearest_customers[i]
            waiting_time = max(time_windows[new_stop, 0] - (current_time + time_matrix[current_stop, new_stop]), 0)
            total_return_time = time_matrix[current_stop, new_stop] + waiting_time + service_times[new_stop] + \
                                time_matrix[new_stop, 0]
            if current_time + time_matrix[current_stop, new_stop] > \
                    time_windows[new_stop, 1] or remaining_capacity < demands[
                new_stop] or new_stop not in unvisited_customers or current_time + total_return_time > \
                    time_windows[0, 1]:
                i += 1
            else:
                current_route.append(new_stop)
                remaining_capacity -= demands[new_stop]
                current_time = max(current_time + time_matrix[current_stop, new_stop],
                                   time_windows[new_stop, 0]) + service_times[new_stop]
                unvisited_customers.remove(new_stop)
                current_stop = new_stop
                feasible_addition = True

            if not feasible_addition and i == num_customers + 1:
                current_route.append(0)
                solution.append(current_route)
                current_stop = 0
                current_route = [0]
                remaining_capacity = truck_capacity
                current_time = 0
                break

    current_route.append(0)
    solution.append(current_route)

    singular_routes = []
    costs = []
    for route in solution:
        singular_routes.append(convert_ordered_route(route, num_customers))
        costs.append(sum(time_matrix[route[i], route[i + 1]] for i in range(len(route) - 1)))
    return singular_routes, costs, solution


def convert_ordered_route(ordered_route, num_customers):
    route = numpy.zeros(num_customers)
    for customer in ordered_route:
        if customer != 0:
            route[customer - 1] = 1
    return route


def create_forbidden_edges_list(num_customers, forbidden_edges, compelled_edges):
    forbid_copy = forbidden_edges.copy()
    for edge in compelled_edges:
        forbid_copy += [[x, edge[1]] for x in range(num_customers + 1) if x != edge[0]]
    return forbid_copy


def check_route_feasibility(route, time_matrix, time_windows, service_times, demands_data, truck_capacity):
    current_time = max(time_matrix[0, route[1]], time_windows[route[1], 0])
    total_capacity = 0

    for i in range(1, len(route)):
        if round(current_time, 3) > time_windows[route[i], 1]:
            print("Time Window violated")
            print(route[i])
            return False
        current_time += service_times[route[i]]
        total_capacity += demands_data[route[i]]
        if round(total_capacity, 3) > truck_capacity:
            print("Truck Capacity Violated")
            print(route[i])
            return False
        if i < len(route) - 1:
            # travel to next node
            current_time += time_matrix[route[i], route[i + 1]]
            current_time = max(current_time, time_windows[route[i + 1], 0])
    return True


def main():
    POMO = True
    heuristic = True
    solomon = False
    num_customers = 100
    file = "config.json"
    with open(file, 'r') as f:
        config = json.load(f)

    data_iterator(config, POMO, num_customers, heuristic, solomon)


if __name__ == "__main__":
    main()
