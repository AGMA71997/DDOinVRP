import pickle
import os
import numpy
import torch
import json


def create_price(time_matrix, duals):
    duals = duals.copy()
    duals = numpy.array(duals)
    duals = duals.reshape((len(duals), 1))
    return (time_matrix - duals) * -1


def data_iterator(config, POMO):
    directory = config["storge_directory_raw"]
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

    num_customers = len(CL[0]) - 1
    if POMO:
        process_data_for_POMO(CL, TML, TWL, DL, STL, VCL, DUL, num_customers, config)
    else:
        os.chdir(config['SB3 Data'])
        pickle_out = open('ESPRCTW_Data' + str(num_customers), 'wb')
        pickle.dump([TML, TWL, DL, STL, VCL, DUL], pickle_out)
        pickle_out.close()


def process_data_for_POMO(CL, TML, TWL, DL, STL, VCL, DUL, num_customers, config):
    depot_CL = []
    depot_TW = []
    PL = []
    max_dual = 0
    for x in range(len(CL)):
        depot_CL.append(CL[x][0, :])
        tw_scaler = TWL[x][0, 1]
        depot_TW.append(TWL[x][0, :] / tw_scaler)
        PL.append(create_price(TML[x], DUL[x]))

        CL[x] = numpy.delete(CL[x], 0, 0)
        TWL[x] = numpy.delete(TWL[x], 0, 0) / tw_scaler
        TML[x] = TML[x] / tw_scaler
        DL[x] = numpy.delete(DL[x], 0, 0) / VCL[x]
        STL[x] = numpy.delete(STL[x], 0, 0) / tw_scaler
        if max(DUL[x]) > max_dual:
            max_dual = max(DUL[x])
            print(max_dual)
        DUL[x] = numpy.delete(DUL[x], 0, 0) / 10
        min_val = numpy.min(PL[x])
        max_val = numpy.max(PL[x])
        PL[x] = PL[x] / max(abs(max_val), abs(min_val))

    depot_CL = torch.tensor(depot_CL, dtype=torch.float32)
    depot_TW = torch.tensor(depot_TW, dtype=torch.float32)
    CL = torch.tensor(CL, dtype=torch.float32)
    TWL = torch.tensor(TWL, dtype=torch.float32)
    TML = torch.tensor(TML, dtype=torch.float32)
    DL = torch.tensor(DL, dtype=torch.float32)
    STL = torch.tensor(STL, dtype=torch.float32)
    DUL = torch.tensor(DUL, dtype=torch.float32)
    PL = torch.tensor(PL, dtype=torch.float32)

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

    os.chdir(config["POMO Data"])
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

    file = "config.json"
    with open(file, 'r') as f:
        config = json.load(f)

    data_iterator(config, POMO)


if __name__ == "__main__":
    main()
