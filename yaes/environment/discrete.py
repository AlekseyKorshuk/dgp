import numpy as np
from gym import Env

from yaes.environment import Environment


class DiscreteEnvironment(Environment):
    def __init__(self, gym_env: Env):
        """
        Discrete Environment wrapper for OpenAI Gym environments

        :param gym_env: gym environment
        """
        super().__init__(gym_env)

    def check_action(self, action):
        """
        Checks if the action is within the action space bounds
        :param action: action to check
        :return: True if action is within bounds, False otherwise
        """
        if isinstance(action, np.ndarray):
            action = int(action)
        assert type(
            action) == int, f"This environment is Discrete. Action must be an integer, got {type(action)} {action}"

    def is_discrete(self):
        """
        Returns True if the environment is discrete, False otherwise
        :return: True if the environment is discrete, False otherwise
        """
        return True
