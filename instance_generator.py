import numpy
import os
import json


class Instance_Generator(object):

    def __init__(self, file_path=None, config=None, N=None):

        if file_path is None:
            self.N = N
            assert N is not None
            self.coords = None
            self.time_matrix = self.create_time_matrix()
            self.time_windows = self.create_time_windows()
            VC_map = {20: 30, 50: 40, 100: 50}
            self.vehicle_capacity = VC_map[self.N]
            self.demands = self.create_demands()
            self.service_times = self.create_service_times()
        else:
            self.generate_instance(file_path, config)

    def generate_instance(self, file_path, config=None, instance_type="Solomon"):

        if instance_type == "Solomon":
            VC_dict = {}
            with open(config["Solomon Dataset"] + "/capacities.txt") as f:
                contents = f.readlines()

            for item in contents:
                string_list = item.split(":")
                key = string_list[0] + string_list[1][0]
                value = string_list[2].replace(r'\n', '')
                VC_dict[key] = int(value)

            with open(file_path) as f:
                contents = f.readlines()
            self.N = len(contents) - 1

            file_name = os.path.basename(file_path)
            try:
                self.vehicle_capacity = VC_dict[file_name[0:2]]
            except:
                self.vehicle_capacity = VC_dict[file_name[0:3]]
            self.coords = numpy.zeros((self.N + 1, 2))
            self.demands = numpy.zeros((self.N + 1))
            self.time_windows = numpy.zeros((self.N + 1, 2))
            self.service_times = numpy.zeros((self.N + 1))

            for index, item in enumerate(contents):
                string_list = item.split()
                self.coords[index, 0] = float(string_list[1])
                self.coords[index, 1] = float(string_list[2])
                self.demands[index] = float(string_list[3])
                self.time_windows[index, 0] = float(string_list[4])
                self.time_windows[index, 1] = float(string_list[5])
                self.service_times[index] = float(string_list[6])

            self.time_matrix = self.create_time_matrix(self.coords)

    def create_time_matrix(self, customer_locations=None):

        if customer_locations is None:
            customer_locations = numpy.random.random_sample((self.N + 1, 2))
            self.coords = customer_locations

        time_matrix = numpy.zeros((self.N + 1, self.N + 1))

        for i in range(self.N + 1):
            for j in range(self.N + 1):
                if i != j:
                    time_matrix[i, j] = numpy.linalg.norm(customer_locations[i, :] - customer_locations[j, :])

        return time_matrix

    def create_time_windows(self, minimum_margin=2):
        time_windows = numpy.zeros((self.N + 1, 2))
        for i in range(self.N + 1):
            if i != 0:
                time_windows[i, 0] = numpy.random.randint(0, 10)
                time_windows[i, 1] = numpy.random.randint(time_windows[i, 0] + minimum_margin, 18)

        time_windows[0, 1] = 18

        return time_windows

    def create_demands(self):
        demands = numpy.zeros(self.N + 1)
        for i in range(self.N + 1):
            if i != 0:
                demands[i] = numpy.random.randint(1, 10)

        return demands

    def create_service_times(self):
        service_times = numpy.zeros(self.N + 1)
        for i in range(self.N + 1):
            if i != 0:
                service_times[i] = numpy.random.uniform(0.2, 0.5)

        return service_times


def main():
    file = "config.json"
    with open(file, 'r') as f:
        config = json.load(f)

    instance = config["Solomon Dataset"] + "/C101.txt"
    VRP_instance = Instance_Generator(instance, config)


if __name__ == "__main__":
    main()
