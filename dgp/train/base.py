from dgp.environment import Environment
from dgp.agent.stable_baselines import RLAgent
from stable_baselines3.common.monitor import Monitor


class Trainer:

    def __init__(self, env: Environment, agent_class, log_dir="tmp/", **config):
        self.env = env
        self.agent_class = agent_class
        self.config = config
        self.log_dir = log_dir

    def train(self):
        self.agent_class.env = self.env
        if isinstance(self.agent_class, RLAgent):
            return self.train_rl()
        else:
            return self.train_ea()

    def train_rl(self):
        best_agent = self.agent_class.train(self.log_dir)
        training_stats = {}
        eval_stats = self.evaluate(best_agent, num_episodes=1)
        return best_agent, training_stats, eval_stats

    def train_ea(self):
        self.env.gym_env = Monitor(self.env.gym_env, self.log_dir)
        best_agent, training_stats = self.agent_class.train()
        eval_stats = self.evaluate(best_agent, num_episodes=1)
        self.env.gym_env = self.env.gym_env.env
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
