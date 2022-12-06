import itertools
import time
from collections import OrderedDict

import gym.spaces
import numpy as np
from gym import Env


class Environment:
    def __init__(self, gym_env: Env):
        """
        Environment wrapper for OpenAI Gym environments

        :param gym_env: gym environment
        """
        self.gym_env = gym_env
        self.state_ = self.gym_env.reset()
        self.done = False
        self.rewards = []
        self.sum_reward = True
        self.info = None
        self.truncated = False

    def reset(self):
        """
        Resets the environment
        """
        self.state_ = self.gym_env.reset()
        self.done = False
        self.rewards = []

    def step(self, action) -> (np.ndarray, float, bool, dict):
        """
        Performs an action in the environment

        :param action: action to perform
        :return: (state, reward, done, info)
        """
        try:
            self.state_, reward, self.done, self.info = self.gym_env.step(action)
        except Exception as e:
            try:
                self.state_, reward, self.done, self.truncated, self.info = self.gym_env.step(action)
            except Exception as e:
                print(self.gym_env.step(action))
                raise e
        self.rewards.append(reward)
        return self.state, self.reward, self.done, self.info

    def render(self):
        """
        Renders the environment
        """
        self.gym_env.render()

    def close(self):
        """
        Closes the environment
        """
        self.gym_env.close()

    def play(self, agent_class, render=False, sleep=1 / 30, max_duration=1) -> dict:
        """
        Plays the environment with an agent

        :param agent_class: agent class
        :param render: whether to render the environment
        :param sleep: time to sleep between steps
        :param max_duration: maximum duration of the episode

        :return: dict with the following
            - reward: total reward
            - steps: number of steps
            - info: info dict
            - agent: agent instance
        """
        if type(agent_class) == type:
            agent = agent_class(self.gym_env.observation_space.shape[0], self.gym_env.action_space.n,
                                is_discrete=True)
        else:
            agent = agent_class
        self.reset()
        start_time = time.time()
        while not (self.done or self.truncated or (time.time() - start_time > max_duration)):
            action = agent.predict(self.state)
            if type(action) == tuple:
                action = action[0]
            self.step(action)
            if render:
                self.render()
                time.sleep(sleep)
        return {
            "reward": self.reward,
            "steps": self.get_elapsed_steps(),
            "agent": agent,
            "info": self.info,
        }

    def demo(self, render=False, sleep=0) -> dict:
        """
        Plays the environment with a random agent

        :param render: whether to render the environment
        :param sleep: time to sleep between steps

        :return: dict with the following
            - reward: total reward
            - steps: number of steps
            - info: info dict
        """
        self.reset()
        while not (self.done or self.truncated):
            action = self.gym_env.action_space.sample()
            if type(action) == tuple:
                action = action[0]
            self.step(action)
            if render:
                self.render()
                time.sleep(sleep)

        return {
            "reward": self.reward,
            "steps": self.get_elapsed_steps(),
            "info": self.info,
        }

    def get_elapsed_steps(self) -> int:
        """
        Returns the number of elapsed steps

        :return: number of elapsed steps
        """
        if "score" in self.info:
            return self.info["score"]
        try:
            return self.gym_env._elapsed_steps
        except:
            return 0

    def get_observation_space(self) -> int:
        """
        Returns the observation space

        :return: observation space
        """
        if type(self.gym_env.observation_space) == gym.spaces.Discrete:
            return 1
        if type(self.gym_env.observation_space) == gym.spaces.dict.Dict:
            num_inputs = 0
            for key in self.gym_env.observation_space.spaces:
                num_inputs += self.gym_env.observation_space.spaces[key].shape[0]
            return num_inputs
        return np.prod(self.gym_env.observation_space.shape)

    def get_action_space(self) -> int:
        """
        Returns the action space

        :return: action space
        """
        if type(self.gym_env.action_space) == gym.spaces.Discrete:
            return self.gym_env.action_space.n
        return self.gym_env.action_space.shape[0]

    def is_discrete(self) -> bool:
        """
        Returns whether the environment is discrete

        :return: whether the environment is discrete
        """
        return type(self.gym_env.action_space) == gym.spaces.Discrete

    @property
    def state(self):
        """
        Returns the current state

        :return: current state
        """
        if type(self.gym_env.observation_space) == gym.spaces.box.Box and type(self.state_) != tuple:
            if type(self.state_[0]) == float or type(self.state_[0]) == np.float64:
                return self.state_
            try:
                state = np.concatenate(self.state_)
                while len(state.shape) > 1:
                    state = np.concatenate(state)
                return list(state)
            except:
                return self.state_

        if type(self.gym_env.observation_space) == gym.spaces.Discrete:
            return list([self.state_])
        if type(self.gym_env.observation_space) == gym.spaces.dict.Dict or type(
                self.gym_env.observation_space) == gym.spaces.box.Box:
            state = []
            if type(self.state_) == list or type(self.state_) == tuple:
                self.state_ = self.state_[0]
            if type(self.state_) == OrderedDict:
                for key in self.state_:
                    state += self.state_[key].tolist()
            else:
                try:
                    state = np.concatenate(self.state_)
                    while len(state.shape) > 1:
                        state = np.concatenate(state)
                    return list(state)
                except:
                    return self.state_
            return list(state)
        if type(self.state_[0]) != list:
            return list(self.state_)
        state = np.concatenate(self.state_)
        while len(state.shape) > 1:
            state = np.concatenate(state)
        return list(state)

    @property
    def reward(self) -> float:
        """
        Returns the current reward

        :return: current reward
        """
        if self.sum_reward:
            return sum(self.rewards)
        return self.rewards[-1]
