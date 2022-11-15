import time


class Environment:
    def __init__(self, gym_env):
        self.gym_env = gym_env
        self.state = self.gym_env.reset()
        self.done = False
        self.reward = None
        self.info = None

    def reset(self):
        self.state = self.gym_env.reset()
        self.done = False
        self.reward = None

    def step(self, action):
        try:
            self.state, self.reward, self.done, self.info = self.gym_env.step(action)
        except Exception as e:
            try:
                self.state, self.reward, self.done, _, self.info = self.gym_env.step(action)
            except Exception as e:
                print(action)
                print(self.gym_env.step(action))
                raise e

        return self.state, self.reward, self.done, self.info

    def render(self):
        self.gym_env.render()

    def close(self):
        self.gym_env.close()

    def play(self, agent_class, render=False, sleep=1 / 30):
        if type(agent_class) == type:
            agent = agent_class(self.gym_env.observation_space.shape[0], self.gym_env.action_space.n,
                                is_discrete=True)
        else:
            agent = agent_class
        self.reset()
        while not self.done:
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

    def get_elapsed_steps(self):
        if "score" in self.info:
            return self.info["score"]
        try:
            return self.gym_env._elapsed_steps
        except:
            return 0

    def get_observation_space(self):
        return self.gym_env.observation_space.shape[0]

    def get_action_space(self):
        return self.gym_env.action_space.n

    def is_discrete(self):
        return True
