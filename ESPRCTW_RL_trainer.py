import gym
import numpy as np

from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

import random
from instance_generator import Instance_Generator
from column_generation import MasterProblem, initialize_columns


def make_env(i, envs_list):
    def _init():
        return envs_list[i]

    return _init


def mask_fn(env):
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    return env.unwrapped.valid_action_mask()


def standardize(matrix, max_val=None):
    min_val = np.min(matrix)
    if max_val is None:
        max_val = np.max(matrix)
    return (matrix - min_val) / (max_val - min_val)


class ESPRCTW_Env(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, num_customers, vehicle_capacity, time_matrix, demands, time_windows, duals,
                 service_times, forbidden_edges):
        super(ESPRCTW_Env, self).__init__()
        self.num_customers = num_customers
        self.vehicle_capacity = vehicle_capacity
        self.time_matrix = time_matrix
        self.demands = demands
        self.time_windows = time_windows
        self.service_times = service_times
        self.forbidden_edges = forbidden_edges
        self.discount_factor = 1

        self.price = self.calculate_price(duals)
        self.original_price = np.copy(self.price)
        self.price = standardize(self.price)

        self.best_reward = 0
        self.K = 5
        self.determine_nearest_customers()

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(num_customers + 1)

        # Example for using image as input:
        self.observation_space = spaces.Box(low=0, high=255, shape=(num_customers + 1, 6), dtype=np.float32)

    def calculate_price(self, duals):
        duals = duals.copy()
        duals.insert(0, 0)
        duals = np.array(duals)
        duals = duals.reshape((len(duals), 1))
        return (self.time_matrix - duals) * -1

    def calculate_real_reward(self, label):
        return sum(self.original_price[label[i], label[i + 1]] for i in range(len(label) - 1))

    def update_instance(self, num_customers, vehicle_capacity, time_matrix, demands, time_windows, duals,
                        service_times, forbidden_edges):
        self.__init__(num_customers, vehicle_capacity, time_matrix, demands, time_windows, duals,
                      service_times, forbidden_edges)

    def determine_nearest_customers(self):
        average_price = np.zeros((self.num_customers + 1))
        average_capacity = np.zeros((self.num_customers + 1))
        average_time = np.zeros(self.num_customers + 1)
        for i in range(len(average_capacity)):
            nearest_customers = np.argsort(self.time_matrix[i, :].copy())[:self.K]
            average_price[i] = sum(self.price[i, int(x)] for x in nearest_customers) / self.K
            average_capacity[i] = sum(self.demands[int(x)] for x in nearest_customers) / self.K
            average_time[i] = sum(self.time_matrix[i, int(x)] for x in nearest_customers) / self.K

        self.average_price = average_price
        self.average_capacity = average_capacity
        self.average_time = average_time

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
        obs = np.zeros((self.num_customers + 1, 6))

        for i in range(len(obs)):
            obs[i, :] = [self.price[self.start_point, i], self.demands[i],
                         self.time_matrix[self.start_point, i] +
                         max(self.time_windows[i, 0] - (self.current_time + self.time_matrix[self.start_point, i]), 0) +
                         self.service_times[i],
                         self.average_price[i],
                         self.average_capacity[i],
                         self.average_time[i]]

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
        if (self.current_label[-1] == 0 and len(self.current_label) > 2):
            done = True

        obs = self._next_observation()
        truncated = done
        if not done:
            reward = 0
        else:
            reward = self.current_price
            if reward > self.best_reward:
                self.best_reward = reward

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

        feasible_actions[0] = True
        return feasible_actions

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        pass


class ESPRCTW_RL_trainer(object):
    pass


def main():
    random.seed(5)
    np.random.seed(25)
    num_customers = 10

    print("This instance has " + str(num_customers) + " customers.")
    VRP_instance = Instance_Generator(num_customers)
    time_matrix = VRP_instance.time_matrix
    time_windows = VRP_instance.time_windows
    demands = VRP_instance.demands
    vehicle_capacity = VRP_instance.vehicle_capacity
    service_times = VRP_instance.service_times

    forbidden_edges = []
    compelled_edges = []

    initial_routes, initial_costs, initial_orders = initialize_columns(num_customers, vehicle_capacity, time_matrix,
                                                                       service_times, time_windows, demands)
    master_problem = MasterProblem(num_customers, initial_routes, initial_costs, initial_orders, forbidden_edges,
                                   compelled_edges)
    master_problem.solve()
    duals = master_problem.retain_duals()

    # Environment wrapper Custom Vectorized Normalized Environment
    # env = VecNormalize(env, norm_obs=True, norm_reward=True)

    env = ESPRCTW_Env(num_customers, vehicle_capacity, time_matrix, demands, time_windows, duals,
                      service_times, forbidden_edges)
    env = ActionMasker(env, mask_fn)  # Maskable environment

    randis = np.random.uniform(low=0.5, high=2.5, size=len(duals))
    duals_2 = [duals[x] - randis[x] for x in range(len(duals))]

    env_2 = ESPRCTW_Env(num_customers, vehicle_capacity, time_matrix, demands, time_windows, duals_2,
                        service_times, forbidden_edges)
    env_2 = ActionMasker(env_2, mask_fn)  # Maskable environment

    envs_list = [env, env_2]

    big_env = DummyVecEnv([make_env(i, envs_list) for i in range(len(envs_list))])

    randis = np.random.uniform(low=0.5, high=2.5, size=len(duals))
    duals_3 = [duals[x] - randis[x] for x in range(len(duals))]

    env_3 = ESPRCTW_Env(num_customers, vehicle_capacity, time_matrix, demands, time_windows, duals_3,
                        service_times, forbidden_edges)
    env_3 = ActionMasker(env_3, mask_fn)

    envs_list_2 = [env_2, env_3]

    big_env_2 = DummyVecEnv([make_env(i, envs_list) for i in range(len(envs_list_2))])

    # model = MaskablePPO.load("PPO maskable RL agent")
    model = MaskablePPO(MaskableActorCriticPolicy, big_env, verbose=1, normalize_advantage=True)

    indices = list(range(1))
    envs = model.get_env()._get_target_envs(indices)
    for env in envs:
        pass

    model.learn(total_timesteps=1000, log_interval=1)
    print("Trained")

    model.set_env(big_env_2)
    model.learn(total_timesteps=1000, log_interval=1)
    print("Trained Again")

    vec_env = DummyVecEnv([lambda: envs[0]])

    print(evaluate_policy(model, vec_env, deterministic=True))
    obs = vec_env.reset()
    label = []
    for i in range(5):
        action_mask = env.unwrapped.valid_action_mask()  # vec_env.env_method("valid_action_mask")

        action, _state = model.predict(obs, action_masks=action_mask, deterministic=True)
        print(action)
        # print(model.policy.get_distribution(torch.from_numpy(obs)).distribution.probs)
        obs, reward, done, info = vec_env.step(action)
        print(env.unwrapped.current_label)

        label.append(int(action))
        # VecEnv resets automatically
        if done:
            print("The real reward is: " + str(env.unwrapped.calculate_real_reward(label)))
            label = []
            obs = vec_env.reset()

    # model.save("PPO maskable RL agent")


if __name__ == "__main__":
    main()
