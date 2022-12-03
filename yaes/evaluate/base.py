import os
from yaes.environment import Environment
from yaes.train import Trainer
import pandas as pd


class Evaluator:
    def __init__(self, env: Environment):
        self.env = env
        self.env.reset()

    def evaluate(self, agents):
        stats = []
        monitor_df_paths = []
        for agent_class in agents:
            log_dir = "monitor_stats_{}".format(agent_class.__class__.__name__)
            os.makedirs(log_dir, exist_ok=True)
            monitor_df_paths.append(log_dir + "/monitor.csv")
            trainer = Trainer(self.env, agent_class, log_dir=log_dir)
            best_agent, training_stats, eval_stats = trainer.train()
            # wait for file to be written
            # shutil.rmtree(log_dir)
            stats.append(
                {
                    "training_stats": training_stats,
                    "eval_stats": eval_stats,
                    "best_agent": best_agent,
                    # "monitor_df": df
                }
            )
        for i, path in enumerate(monitor_df_paths):
            df = pd.read_csv(path, header=1)
            stats[i]["monitor_df"] = df
            # shutil.rmtree(path.split("/")[0])
        return stats
