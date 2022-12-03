import numpy as np

from yaes.environment import Environment


class ContinuousEnvironment(Environment):
    def __init__(self, gym_env):
        super().__init__(gym_env)

    def get_bounds(self):
        low = self.gym_env.action_space.low
        if type(low) == np.ndarray or type(low) == list:
            low = low[0]
        high = self.gym_env.action_space.high
        if type(high) == np.ndarray or type(high) == list:
            high = high[0]
        return low, high

    def check_action(self, action):
        assert type(
            action) == list, f"This environment is Continuous. Action must be a list, got {type(action)}, {action}"
        bounds = self.get_bounds()
        for a in action:
            assert type(a) == float, f"This environment is Discrete. Action must be an integer, got {type(a)}, {a}"
            assert bounds[0] <= a <= bounds[1], f"Action {a} is out of bounds {bounds}"

    def is_discrete(self):
        return False
