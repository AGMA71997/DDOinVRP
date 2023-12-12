import os

import gym
import numpy as np
import json

from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
import pickle


def make_env(i, envs_list):
    def _init():
        return envs_list[i]

    return _init


def calculate_price(time_matrix, duals):
    duals = duals.copy()
    duals.insert(0, 0)
    duals = np.array(duals)
    duals = duals.reshape((len(duals), 1))
    return (np.copy(time_matrix) - duals) * -1


def mask_fn(env):
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    return env.unwrapped.valid_action_mask()


def standardize(matrix):
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    return (matrix - min_val) / (max_val - min_val)


class ESPRCTW_Env(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, num_customers, vehicle_capacity, time_matrix, demands, time_windows, duals,
                 service_times, forbidden_edges):
        super(ESPRCTW_Env, self).__init__()

        tw_scalar = time_windows[0, 1]
        self.price = calculate_price(time_matrix, duals)
        self.original_price = np.copy(self.price)
        # self.price = standardize(self.price)

        self.num_customers = num_customers
        self.vehicle_capacity = 1
        self.time_matrix = time_matrix / tw_scalar
        self.demands = demands / vehicle_capacity
        self.time_windows = time_windows / tw_scalar
        self.service_times = service_times / tw_scalar
        self.forbidden_edges = forbidden_edges
        self.edge_performance = {}

        self.current_time = None
        self.current_step = None
        self.current_price = None
        self.start_point = None
        self.current_label = None
        self.remaining_capacity = self.vehicle_capacity
        self.mask = None

        #self.best_reward = 0
        self.K = min(max(5, int(num_customers / 10)), 20)

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(num_customers + 1)

        # Example for using image as input:
        self.observation_space = spaces.Box(low=0, high=255, shape=(num_customers + 2, 3), dtype=np.float32)

    def calculate_real_reward(self, label):
        return sum(self.original_price[label[i], label[i + 1]] for i in range(len(label) - 1))

    def determine_nearest_customers(self):
        average_price = np.zeros((self.num_customers + 1))
        average_capacity = np.zeros((self.num_customers + 1))
        average_time = np.zeros(self.num_customers + 1)
        for i in range(len(average_capacity)):
            if self.mask[i]:
                nearest_customers = np.argsort(self.time_matrix[i, :].copy())[:self.K]
                average_price[i] = sum(self.price[i, int(x)] for x in nearest_customers if
                                       (self.mask[x] and (i, int(x)) not in self.forbidden_edges)) / self.K
                average_capacity[i] = sum(self.demands[int(x)] for x in nearest_customers if
                                          (self.mask[x] and (i, int(x)) not in self.forbidden_edges)) / self.K
                average_time[i] = sum(
                    self.time_matrix[i, int(x)] for x in nearest_customers if
                    (self.mask[x] and (i, int(x)) not in self.forbidden_edges)) / self.K

        return average_price, average_capacity, average_time

    def reset(self, seed, options):
        # Reset the state of the environment to an initial state
        self.start_point = 0
        self.current_label = [0]
        self.remaining_capacity = self.vehicle_capacity
        self.current_time = 0
        self.current_step = 0
        self.current_price = 0

        return self._next_observation(), {}

    def _next_observation(self):
        obs = np.zeros((self.num_customers + 2, 3))
        self.mask = self.valid_action_mask()
        average_price, average_capacity, average_time = self.determine_nearest_customers()
        for i in range(len(obs) - 1):
            if self.mask[i]:
                if (self.start_point, i, len(self.current_label)) in self.edge_performance:
                    count = self.edge_performance[self.start_point, i, len(self.current_label)]
                else:
                    count = 0

                obs[i, :] = [self.price[self.start_point, i], self.demands[i],
                             (self.time_matrix[self.start_point, i] +
                              max(self.time_windows[i, 0] - (self.current_time + self.time_matrix[self.start_point, i]),
                                  0) +
                              self.service_times[i])]

        obs[-1, :] = [self.current_price, 1 - self.remaining_capacity, 1-self.current_time]
        return obs

    def _take_action(self, action):
        self.current_label.append(action)
        self.current_price += self.price[self.start_point, action]
        self.current_time = max(self.current_time + self.time_matrix[self.start_point, action],
                                self.time_windows[action, 0])
        self.current_time += self.service_times[action]
        self.remaining_capacity -= self.demands[action]
        self.start_point = action

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)
        # Consider adding discounted rewards

        self.current_step += 1
        done = False
        if self.current_label[-1] == 0 and len(self.current_label) > 2:
            done = True

        obs = self._next_observation()
        truncated = done
        if not done:
            reward = 0
        else:
            reward = self.current_price

        return obs, reward, done, truncated, {}

    def valid_action_mask(self):
        feasible_actions = np.ones(self.num_customers + 1, dtype=bool)
        for i in range(self.num_customers + 1):
            waiting_time = max(self.time_windows[i, 0] - (self.current_time + self.time_matrix[self.start_point, i]), 0)
            total_return_time = self.time_matrix[self.start_point, i] + waiting_time + self.service_times[i] + \
                                self.time_matrix[i, 0]
            if (self.current_time + self.time_matrix[self.start_point, i] > self.time_windows[
                i, 1] or self.remaining_capacity < self.demands[i] or i in self.current_label
                    or self.current_time + total_return_time > self.time_windows[0, 1] or
                    [self.start_point, i] in self.forbidden_edges):
                feasible_actions[i] = False

        if self.start_point != 0:
            feasible_actions[0] = True
        else:
            feasible_actions[0] = False
        return feasible_actions

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        pass


class ESPRCTW_RL_trainer(object):

    def __init__(self, no_of_epochs, number_of_steps, no_of_envs, load_data, num_customers, config):
        self.data_index = None
        self.model = None
        self.config = config
        self.TML, self.TWL, self.DL, self.STL, self.VCL, self.DUL = None, None, None, None, None, None
        self.num_customers = num_customers
        self.generate_data(load_data)
        self.no_of_epochs = no_of_epochs
        self.no_of_steps = number_of_steps
        self.no_of_envs = no_of_envs

    def generate_data(self, load_data):
        if load_data:
            os.chdir(self.config["SB3 Data"])
            pickle_in = open('ESPRCTW_Data' + str(self.num_customers), 'rb')
            self.TML, self.TWL, self.DL, self.STL, self.VCL, self.DUL = pickle.load(pickle_in)
            self.data_index = 0
        else:
            pass

    def run(self):
        os.chdir(self.config["Saved SB3 Model"])

        for epoch in range(1, self.no_of_epochs + 1):
            self.train_multiple_envs()
            print("Epoch " + str(epoch) + " complete")
            if epoch % 10 == 0:
                self.model.save('ESPRCTW_Solver_' + str(self.num_customers) + "_" + str(epoch))

    def train_multiple_envs(self):
        TML_n = self.TML[self.data_index:self.data_index + self.no_of_envs]
        TWL_n = self.TWL[self.data_index:self.data_index + self.no_of_envs]
        DL_n = self.DL[self.data_index:self.data_index + self.no_of_envs]
        STL_n = self.STL[self.data_index:self.data_index + self.no_of_envs]
        VCL_n = self.VCL[self.data_index:self.data_index + self.no_of_envs]
        DUL_n = self.DUL[self.data_index:self.data_index + self.no_of_envs]
        env_list = []
        for x in range(self.no_of_envs):
            time_matrix, time_windows, demands, service_times, vehicle_capacity, duals = \
                TML_n[x], TWL_n[x], DL_n[x], STL_n[x], VCL_n[x], DUL_n[x]
            env = ESPRCTW_Env(self.num_customers, vehicle_capacity, time_matrix, demands, time_windows, duals[1:],
                              service_times, [])
            env = ActionMasker(env, mask_fn)
            env_list.append(env)

        vec_env = DummyVecEnv([make_env(i, env_list) for i in range(len(env_list))])
        if self.model is None:
            self.model = MaskablePPO(MaskableActorCriticPolicy, vec_env, verbose=1, )
        else:
            self.model.set_env(vec_env)

        self.model.learn(total_timesteps=self.no_of_steps, log_interval=100000)
        self.data_index += self.no_of_envs


def main():
    file = "config.json"
    with open(file, 'r') as f:
        config = json.load(f)

    num_customers = 20
    no_of_epochs = 30
    no_of_steps = num_customers * 500
    no_of_envs = 10
    load_data = True
    trainer = ESPRCTW_RL_trainer(no_of_epochs, no_of_steps, no_of_envs, load_data, num_customers, config)
    trainer.run()


if __name__ == "__main__":
    main()
