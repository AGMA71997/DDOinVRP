import gym
import sys
from gym import spaces
from stable_baselines3 import PPO
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper
from stable_baselines3.common.vec_env import DummyVecEnv

import random
from instance_generator import Instance_Generator
from column_generation import MasterProblem, initialize_columns


class ESPRCTW_Env(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, num_customers, vehicle_capacity, time_matrix, demands, time_windows, time_limit, duals,
                 service_times, forbidden_edges):
        super(ESPRCTW_Env, self).__init__()
        self.num_customers = num_customers
        self.vehicle_capacity = vehicle_capacity
        self.time_matrix = time_matrix
        self.demands = demands
        self.time_windows = time_windows
        self.time_limit = time_limit
        self.duals = duals
        self.service_times = service_times
        self.price = np.zeros((num_customers + 1, num_customers + 1))
        self.forbidden_edges = forbidden_edges
        self.discount_factor = 0.9

        for i in range(num_customers + 1):
            for j in range(num_customers + 1):
                if i != j:
                    if i != 0:
                        self.price[i, j] = (time_matrix[i, j] - duals[i - 1])*-1
                    else:
                        self.price[i, j] = time_matrix[i, j]

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(num_customers + 1)
        # Example for using image as input:
        self.observation_space = spaces.Box(low=0, high=255, shape=
        (num_customers + 1, 6), dtype=np.float32)

    def reset(self):
        # Reset the state of the environment to an initial state
        self.start_point = 0
        self.current_label = [0]
        self.remaining_capacity = self.vehicle_capacity
        self.current_time = 0
        self.current_step = 0
        self.current_price = 0

        return self._next_observation()

    def _next_observation(self):
        obs = np.zeros((self.num_customers + 1, 6))
        feasible_actions = np.ones(self.num_customers + 1, dtype=bool)
        for i in range(len(obs)):
            total_return_time = self.time_matrix[self.start_point, i] + self.service_times[i] + self.time_matrix[i, 0]
            if (self.current_time + self.time_matrix[self.start_point, i] > self.time_windows[
                i, 1] or self.remaining_capacity < self.demands[i] or i in self.current_label
                    or self.current_time + total_return_time > self.time_windows[0, 1] or
                    [self.start_point, i] in self.forbidden_edges):
                feasible_actions[i] = False
            else:
                obs[i, :] = [self.price[self.start_point, i], self.demands[i],
                             self.time_windows[i, 0] - (self.current_time + self.time_matrix[self.start_point, i]),
                             self.time_matrix[self.start_point, i],
                             self.service_times[i],
                             self.time_windows[i, 1] - self.current_time]
        return obs  # , feasible_actions

    def _take_action(self, action):
        reward = self.price[self.start_point, action]
        self.current_label.append(action)
        self.current_price += reward
        self.current_time += self.time_matrix[self.start_point, action]
        self.current_time = max(self.time_windows[action, 0], self.current_time)
        self.remaining_capacity -= self.demands[action]
        self.start_point = action

        return reward

    def step(self, action):
        # Execute one time step within the environment
        reward = self._take_action(action)
        # Consider adding discounted rewards

        self.current_step += 1
        done = False
        if self.current_label[-1] == 0 and len(self.current_label) > 2:
            done = True
            self.current_step = 0

        obs = self._next_observation()
        return obs, reward, done, {}

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        pass


class VecExtractDictObs(VecEnvWrapper):  # Example environment from stable-baselines3.
    """
    A vectorized wrapper for filtering a specific key from dictionary observations.
    Similar to Gym's FilterObservation wrapper:
        https://github.com/openai/gym/blob/master/gym/wrappers/filter_observation.py

    :param venv: The vectorized environment
    :param key: The key of the dictionary observation
    """

    def __init__(self, venv: VecEnv, key: str):
        self.key = key
        super().__init__(venv=venv, observation_space=venv.observation_space.spaces[self.key])

    def reset(self) -> np.ndarray:
        obs = self.venv.reset()
        return obs[self.key]

    def step_async(self, actions: np.ndarray) -> None:
        self.venv.step_async(actions)

    def step_wait(self) -> VecEnvStepReturn:
        obs, reward, done, info = self.venv.step_wait()
        return obs[self.key], reward, done, info


def main():
    random.seed(5)
    np.random.seed(25)
    num_customers = 12
    VRP_instance = Instance_Generator(num_customers)
    time_matrix = VRP_instance.time_matrix
    time_windows = VRP_instance.time_windows
    demands = VRP_instance.demands
    vehicle_capacity = VRP_instance.vehicle_capacity
    time_limit = VRP_instance.time_limit
    service_times = VRP_instance.service_times
    forbidden_edges = []
    compelled_edges = []

    initial_routes, initial_costs, initial_orders = initialize_columns(num_customers, time_matrix)
    master_problem = MasterProblem(num_customers, initial_routes, initial_costs, initial_orders, forbidden_edges,
                                   compelled_edges)
    master_problem.solve()
    duals = master_problem.retain_duals()

    # Example deployment of environment with stable baseline
    # env = gym.make("CartPole-v1", render_mode="rgb_array")  # Random registered environment

    env = DummyVecEnv(
        [lambda: ESPRCTW_Env(num_customers, vehicle_capacity, time_matrix, demands, time_windows, time_limit, duals,
                             service_times, forbidden_edges)])  # Custom Vectorized Dummy Environment

    # Wrap the DummyVecEnv
    # env = VecExtractDictObs(env, key="observation")

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=20000)

    vec_env = model.get_env()
    obs = vec_env.reset()
    for i in range(25):
        action, _state = model.predict(obs)
        obs, reward, is_route, info = vec_env.step(action)
        print(action)
        print(vec_env.get_attr("current_label"))
        # vec_env.render("human")

        # VecEnv resets automatically
        # if done:
        # obs = vec_env.reset()


if __name__ == "__main__":
    main()
