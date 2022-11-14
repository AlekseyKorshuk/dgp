import random
from yaes.environment import Environment


class Agent:
    def __init__(self, env: Environment):
        self.env = env
        self.num_states = self.env.get_observation_space()
        self.num_actions = self.env.get_action_space()
        self.is_discrete = self.env.is_discrete()

    def predict(self, state):
        if self.is_discrete:
            action = random.randint(0, self.num_actions - 1)
        else:
            action = random.uniform(0, 1)
        return action

    # def train(self):
    #     trainer = Trainer(self.env, self.agent_class)
