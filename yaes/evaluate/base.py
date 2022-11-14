from yaes.environment import Environment
from yaes.train import Trainer


class Evaluator:
    def __init__(self, env: Environment):
        self.env = env
        self.env.reset()

    def evaluate(self, agents):
        stats = []
        for agent_class in agents:
            trainer = Trainer(self.env, agent_class)
            best_agent, training_stats, eval_stats = trainer.train()
            stats.append(
                {
                    "training_stats": training_stats,
                    "eval_stats": eval_stats,
                    "best_agent": best_agent,
                }
            )
        return stats
