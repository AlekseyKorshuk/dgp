import random
import uuid

from .base import Agent, AgentHelper
from .deap_primitives import basic_primitive_set
from deap import gp, creator, tools


class ModiAgent(Agent):
    def __init__(self, env):
        """
        This class contains declarations required for evolution of Modi individuals.
        :param env: an OpenAI Gym environment.
        """
        super().__init__(env)
        creator.create("ModiIndividual", gp.MultiOutputTree, fitness=creator.FitnessMax, pset=self.pset,
                       num_outputs=self.num_actions)
        self.toolbox.register("individual", tools.initIterate, creator.ModiIndividual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

    def _get_agent_helper(self, func) -> AgentHelper:
        """
        Returns an AgentHelper object for the given function.
        :param func: function to be used for prediction.
        """
        formula = str(func)
        func = gp.compile(func, self.pset)
        return super()._get_agent_helper(func, formula=formula)

    def _create_primitive_set(self, num_inputs, num_outputs):
        """
        Creates a primitive set for the given number of inputs and outputs.
        :param num_inputs: number of inputs.
        :param num_outputs: number of outputs.
        """
        random_uuid = uuid.uuid4().hex
        pset = basic_primitive_set(num_inputs, random_uuid)

        for i in range(num_outputs):
            modi_i = gp.Modi(i)
            pset.addPrimitive(modi_i, [float], name=str(modi_i), ret_type=float)

        return pset
