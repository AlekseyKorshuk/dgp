from yaes.environment import Environment


class ContinuousEnvironment(Environment):
    def __init__(self, gym_env):
        super().__init__(gym_env)

    def get_bounds(self):
        return self.gym_env.action_space.low, self.gym_env.action_space.high

    def check_action(self, action):
        assert type(action) == list, f"This environment is Discrete. Action must be an integer, got {type(action)}"
        bounds = self.get_bounds()
        for a in action:
            assert type(a) == float, f"This environment is Discrete. Action must be an integer, got {type(a)}"
            assert bounds[0] <= a <= bounds[1], f"Action {a} is out of bounds {bounds}"

    def is_discrete(self):
        return False
