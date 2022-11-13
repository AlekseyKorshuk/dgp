import random


class Agent:
    def __init__(self, num_states, num_actions, is_discrete=True):
        self.num_states = num_states
        self.num_actions = num_actions
        self.is_discrete = is_discrete

    def predict(self, state):
        if self.is_discrete:
            action = random.randint(0, self.num_actions - 1)
        else:
            action = random.uniform(0, 1)
        return action

    # def train(self):
    #     trainer = Trainer(self.env, self.agent_class)
