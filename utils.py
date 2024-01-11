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


def main():
    POMO = True

    file = "config.json"
    with open(file, 'r') as f:
        config = json.load(f)

    data_iterator(config, POMO)


if __name__ == "__main__":
    main()
