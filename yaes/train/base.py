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
        eval_stats = self.evaluate(best_agent, num_episodes=1)
        return best_agent, training_stats, eval_stats

    def train_ea(self):
        self.env.reset()
        best_agent, training_stats = self.agent_class.train()
        # TODO: Implement this method
        eval_stats = self.evaluate(best_agent, num_episodes=1)
        return best_agent, training_stats, eval_stats

    def evaluate(self, agent, num_episodes=1):
        eval_stats = []
        for _ in range(num_episodes):
            self.env.reset()
            stats = self.env.play(agent, render=False)
            eval_stats.append(stats)
        stats = {
            "reward": sum([s["reward"] if s["reward"] is not None else 0 for s in eval_stats]) / len(eval_stats),
            "steps": sum([s["steps"] if s["reward"] is not None else 0 for s in eval_stats]) / len(eval_stats),
        }
        return stats
