import dill
import gym
from yaes.environment import wrap_env
import sys
import gym_pygame


def visualize_results(gym_name: str,
                  path_to_best_agent: str = "best_agent.pkl",
                  videos_dir: str = "agent_preview_copter",
                  max_duration: int = 100):
    """
    Render a play of the best agent.

    :param gym_name: Name of the gym.
    :param path_to_best_agent: location of the pickled best agent
    :param videos_dir: directory in rendered video will be stored
    :param max_duration: maximum duration of video in seconds
    """
    gym_env = gym.make(gym_name)
    env = wrap_env(gym_env)

    with open(path_to_best_agent, 'rb') as f:
        best_agent = dill.load(f, fix_imports=False, encoding="ASCII", errors="")

    env.gym_env = gym.wrappers.RecordVideo(env.gym_env, videos_dir)
    print(env.play(best_agent, False, max_duration=max_duration))


if __name__ == '__main__':
    visualize_results(sys.argv[1], sys.argv[2])
