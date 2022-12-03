from .base import Agent
from .deap_primitives import basic_primitive_set
from deap import gp, creator, tools


class ModiAgent(Agent):
    def __init__(self, env):
        super().__init__(env)
        creator.create("ModiIndividual", gp.MultiOutputTree, fitness=creator.FitnessMax, pset=self.pset,
                       num_outputs=self.num_actions)
        self.toolbox.register("individual", tools.initIterate, creator.ModiIndividual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

    def _get_agent_helper(self, func):
        func = gp.compile(func, self.pset)
        return super()._get_agent_helper(func)

    def _create_primitive_set(self, num_inputs, num_outputs):
        pset = basic_primitive_set(num_inputs, "modi_pset")

        for i in range(num_outputs):
            modi_i = gp.Modi(i)
            pset.addPrimitive(modi_i, [float], name=str(modi_i), ret_type=float)

        return pset
