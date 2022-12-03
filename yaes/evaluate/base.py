import os
import shutil
from copy import deepcopy



from yaes.environment import Environment
from yaes.train import Trainer
import pandas as pd


class Evaluator:
    def __init__(self, env: Environment):
        self.env = env
        self.env.reset()

    def evaluate(self, agents):
        stats = []
        for agent_class in agents:
            log_dir = "monitor_stats_{}".format(agent_class.__str__())
            os.makedirs(log_dir, exist_ok=True)
            trainer = Trainer(self.env, agent_class, log_dir=log_dir)
            best_agent, training_stats, eval_stats = trainer.train()
            df = pd.read_csv(os.path.join(log_dir, 'monitor.csv'), header=1)
            # shutil.rmtree(log_dir)
            stats.append(
                {
                    "training_stats": training_stats,
                    "eval_stats": eval_stats,
                    "best_agent": best_agent,
                    "monitor_df": df
                }
            )
        return stats
