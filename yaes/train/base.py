from yaes.environment import Environment
from yaes.agent.stable_baselines import RLAgent


class Trainer:

    def __init__(self, env: Environment, agent_class, **config):
        self.env = env
        self.agent_class = agent_class
        self.config = config

    def train(self):
        if isinstance(self.agent_class, RLAgent):
            return self.train_rl()
        else:
            return self.train_ea()

    def train_rl(self):
        best_agent = self.agent_class.train()
        training_stats = {}
        eval_stats = self.evaluate(best_agent, num_episodes=100)
        return best_agent, training_stats, eval_stats

    def train_ea(self):
        training_stats = {
            "reward": 0,
        }
        self.env.reset()
        # TODO: Implement this method
        best_agent = self.agent_class(self.env.gym_env.observation_space.shape[0], self.env.gym_env.action_space.n,
                                      is_discrete=True)
        eval_stats = self.evaluate(best_agent, num_episodes=100)
        return best_agent, training_stats, eval_stats

    def evaluate(self, agent, num_episodes=100):
        eval_stats = []
        for _ in range(num_episodes):
            self.env.reset()
            stats = self.env.play(agent, render=False)
            eval_stats.append(stats)
        stats = {
            "reward": sum([s["reward"] for s in eval_stats]) / len(eval_stats),
            "steps": sum([s["steps"] for s in eval_stats]) / len(eval_stats),
        }
        return stats
