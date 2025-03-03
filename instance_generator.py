import math
import sys
import numpy
import os
import json
from scipy.spatial import distance_matrix


class Instance_Generator(object):

    def __init__(self, file_path=None, config=None, N=None, instance_type="Solomon"):

        if file_path is None:
            self.N = N
            assert N is not None
            self.coords = None
            self.time_matrix = self.create_time_matrix()
            self.time_windows = self.create_time_windows(self.time_matrix)
            VC_map = {20: 30, 30: 30, 40: 30, 50: 40, 100: 50, 200: 50, 400: 100, 500: 50, 600: 120, 800: 150, 1000: 80}
            self.vehicle_capacity = VC_map[self.N]
            self.demands = self.create_demands()
            self.service_times = self.create_service_times()
        else:
            self.generate_instance(instance_type, file_path, config)

    def generate_instance(self, instance_type, file_path, config):

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

            self.time_matrix = distance_matrix(self.coords, self.coords)
        elif instance_type == "G&H":
            with open(file_path, "r") as file:
                lines = file.readlines()

            start_index = None
            for i, line in enumerate(lines):
                if line.strip().startswith("VEHICLE"):
                    start_index = i + 2  # Skip the header line
                    break

            # Extract vehicle numbers and capacities
            vehicle_data = []
            if start_index is not None:
                for line in lines[start_index:]:
                    if line.strip():
                        vehicle_data.append([int(x) for x in line.split()])
                    else:
                        break  # Stop when an empty line is encountered

            # Convert to NumPy array
            vehicle_data_np = numpy.array(vehicle_data)

            # Extract columns
            self.vehicle_number = vehicle_data_np[:, 0].astype(int)[0]
            self.vehicle_capacity = vehicle_data_np[:, 1].astype(int)[0]

            # Find the starting index of the customer data
            start_index = None
            for i, line in enumerate(lines):
                if line.strip().startswith("CUSTOMER"):
                    start_index = i + 2  # Skip header line
                    break

            # Extract customer data
            customer_data = []
            for line in lines[start_index:]:
                if line.strip():
                    customer_data.append([float(x) for x in line.split()])

            # Convert to NumPy arrays
            customer_data_np = numpy.array(customer_data)

            # Extract columns as numpy arrays
            customer_no = customer_data_np[:, 0].astype(int)  # Convert to int for IDs
            self.N = len(customer_no) - 1
            x_coord = customer_data_np[:, 1].reshape(-1, 1)
            y_coord = customer_data_np[:, 2].reshape(-1, 1)
            self.coords = numpy.concatenate((x_coord, y_coord), axis=1)
            self.demands = customer_data_np[:, 3].astype(float)
            ready_time = customer_data_np[:, 4].astype(float).reshape(-1, 1)
            due_date = customer_data_np[:, 5].astype(float).reshape(-1, 1)
            self.time_windows = numpy.concatenate((ready_time, due_date), axis=1)
            self.service_times = customer_data_np[:, 6].astype(float)
            self.time_matrix = distance_matrix(self.coords, self.coords)
        else:
            print("Instance type not recognized")
            sys.exit(0)

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

    def create_time_windows(self, time_matrix):
        time_windows = numpy.zeros((self.N + 1, 2))
        for i in range(self.N + 1):
            if i != 0:
                time_windows[i, 0] = numpy.random.randint(max(math.floor(time_matrix[0, i]), 0), 17)
                tw_width = numpy.random.randint(2, 9)
                time_windows[i, 1] = min(time_windows[i, 0] + tw_width, 18)

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
    num_customers = 1000
    instance_name = "C1_10_1.txt"
    file = "config.json"
    with open(file, 'r') as f:
        config = json.load(f)

    instance = config["G&H Dataset"] + str(num_customers) + "/" + instance_name
    VRP_instance = Instance_Generator(instance, config, instance_type="G&H")


if __name__ == "__main__":
    main()
