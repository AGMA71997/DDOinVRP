import gym
import sys
import torch
import numpy as np

from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper
from stable_baselines3.common.vec_env import DummyVecEnv

from sb3_contrib.common.envs import InvalidActionEnvDiscrete
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.distributions import make_masked_proba_distribution
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

import random
from instance_generator import Instance_Generator
from column_generation import MasterProblem, initialize_columns


def mask_fn(env):
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    return env.valid_action_mask()


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
        self.service_times = service_times
        self.price = np.zeros((num_customers + 1, num_customers + 1))
        self.forbidden_edges = forbidden_edges
        self.discount_factor = 0.9
        self.calculate_price(duals)

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(num_customers + 1)

        # Example for using image as input:
        self.observation_space = spaces.Box(low=0, high=255, shape=(num_customers + 1, 6), dtype=np.float32)

    def calculate_price(self, duals):
        duals.insert(0, 0)
        duals = np.array(duals)
        duals = duals.reshape((len(duals), 1))
        self.price = (self.time_matrix - duals)*-1

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
                         self.time_windows[i, 0] - (self.current_time + self.time_matrix[self.start_point, i]),
                         self.time_matrix[self.start_point, i],
                         self.service_times[i],
                         self.time_windows[i, 1] - self.current_time]
        return obs

    def _take_action(self, action):
        self.current_label.append(action)
        self.current_price += self.price[self.start_point, action]
        self.current_time = max(self.current_time + self.time_matrix[self.start_point, action],
                                self.time_windows[action, 0])
        self.current_time += self.service_times[action]
        self.current_time = max(self.time_windows[action, 0], self.current_time)
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
            self.current_step = 0

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

        feasible_actions[0] = True
        return feasible_actions

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
    num_customers = 30
    VRP_instance = Instance_Generator(num_customers)
    time_matrix = VRP_instance.time_matrix
    time_windows = VRP_instance.time_windows
    demands = VRP_instance.demands
    vehicle_capacity = VRP_instance.vehicle_capacity
    time_limit = VRP_instance.time_limit
    service_times = VRP_instance.service_times
    forbidden_edges = []
    compelled_edges = []

    initial_routes, initial_costs, initial_orders = initialize_columns(num_customers, vehicle_capacity, time_matrix,
                                                                       service_times, time_windows, demands)
    master_problem = MasterProblem(num_customers, initial_routes, initial_costs, initial_orders, forbidden_edges,
                                   compelled_edges)
    master_problem.solve()
    duals = master_problem.retain_duals()

    # Example deployment of environment with stable baseline
    # env = gym.make("CartPole-v1", render_mode="rgb_array")  # Random registered environment

    randis = np.random.uniform(low=0.5, high=2.5, size=len(duals))
    duals_2 = [duals[x] - randis[x] for x in range(len(duals))]

    env = ESPRCTW_Env(num_customers, vehicle_capacity, time_matrix, demands, time_windows, time_limit, duals,
                      service_times, forbidden_edges)

    # env_2 = ESPRCTW_Env(num_customers, vehicle_capacity, time_matrix, demands, time_windows, time_limit, duals_2,
    # service_times, forbidden_edges)

    # Environment wrapper Custom Vectorized Dummy Environment
    # env = DummyVecEnv([lambda: env, lambda:env_2])

    # Wrap the DummyVecEnv
    # env = VecExtractDictObs(env, key="observation")

    env = ActionMasker(env, mask_fn)  # Maskable environment

    model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1)
    # model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=50000)
    # model = MaskablePPO.load("PPO maskable RL agent")

    print(evaluate_policy(model, env, deterministic=True))
    vec_env = model.get_env()
    obs = vec_env.reset()
    for i in range(10):
        action_mask = vec_env.env_method("valid_action_mask")

        action, _state = model.predict(obs, action_masks=action_mask, deterministic=True)
        print(action)
        # print(model.policy.get_distribution(torch.from_numpy(obs)).distribution.probs)
        obs, reward, done, info = vec_env.step(action)
        print(vec_env.get_attr("current_label"))

        # vec_env.render("human")
        # VecEnv resets automatically
        if done:
            obs = vec_env.reset()

    # model.save("PPO maskable RL agent")
    # vec_env.env_method("calculate_price", duals_2)
    # model.learn(total_timesteps=10000)


if __name__ == "__main__":
    main()
