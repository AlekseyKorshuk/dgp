from yaes.environment import Environment
from deap import gp
from deap import creator, base, tools, algorithms
import operator
import numpy as np
from typing import Callable, Union, List, Tuple, Dict, Any


class AgentHelper:
    def __init__(self, func: Callable[[float], List[float]],
                 bounds: Tuple[float, float] = None,
                 is_continuous: bool = False,
                 formula: str = None):
        """
        This class encapsulates postprocessing logic for vector of outputs
        and exposes 'predict' method which will be used by OpenAI Gym.

        :param func: function which accepts state as an input and returns a vector of outputs with scores for each
                     action.
        :param bounds: domain bounds for continuous outputs.
        :param is_continuous: a flag which indicates whether the output should be continuous or discrete.
        :param formula: formula which was used to generate the function.
        """
        self.func = func
        self.bounds = bounds
        self.is_continuous = is_continuous
        self.formula = formula

    def predict(self, state: List[float]) -> Union[List[float], int]:
        """
        Returns the next action (either its index or a magnitude) based on the game state.

        :param state: state of the game.
        :return: action.
        """
        state = list(map(float, state))
        output = self.func(*state)

        if self.is_continuous and self.bounds is not None:
            output = np.clip(output, *self.bounds).tolist()
        else:
            output = int(np.argmax(output))

        return output


def _create_stats():
    """
    Creates a dictionary of statistics for the evolution process.
    """
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    return stats


class Agent:
    def __init__(self, env: Environment):
        """
        This class contains a default set of operations and primitives required for training.

        :param env: an OpenAI Gym environment.
        """
        self.env = env
        self.num_states = self.env.get_observation_space()
        self.num_actions = self.env.get_action_space()
        self.is_discrete = self.env.is_discrete()

        self.pset = self._create_primitive_set(self.num_states, self.num_actions)
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=self.pset,
                       num_outputs=self.num_actions)

        self.agent_helper = self._get_agent_helper

        self.toolbox = self._create_toolbox()
        self.stats = _create_stats()

    def _create_toolbox(self):
        """
        Creates a toolbox for the evolution process.

        :return: toolbox.
        """
        toolbox = base.Toolbox()
        toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=2, max_=3)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register('evaluate', self._fitness)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=self.pset)

        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
        toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

        return toolbox

    def _create_primitive_set(self, num_inputs, num_outputs):
        """
        Creates a primitive set for the evolution process.

        :param num_inputs: number of inputs.
        :param num_outputs: number of outputs.
        """
        raise NotImplementedError

    def _get_agent_helper(self, func, formula=None) -> AgentHelper:
        """
        Returns an agent helper which will be used by OpenAI Gym.

        :param func: function which accepts state as an input and returns a vector of outputs with scores for each
                        action.
        :param formula: formula which was used to generate the function.
        :return: agent helper.
        """
        if self.is_discrete:
            return AgentHelper(func, formula=formula)
        else:
            return AgentHelper(func, bounds=self.env.get_bounds(), is_continuous=True, formula=formula)

    def _fitness(self, agent):
        """
        Calculates the fitness of the agent.

        :param agent: an agent.
        :return: fitness.
        """
        result = self.env.play(self.agent_helper(agent), render=False)
        reward, steps = result["reward"], result["steps"]
        return reward,  # , steps

    def train(self, n_pop: int = 30, cxpb: float = 0.9, mutpb: float = 0.5, n_gens: int = 10):
        """
        This function evolves a population of functions and returns the best performing one.

        :param n_pop: number of individuals in a population.
        :param cxpb: probability of crossover.
        :param mutpb: probability of mutation.
        :param n_gens: number of generations.
        :return: function with the highest fitness
        """
        pop = self.toolbox.population(n=n_pop)
        hof = tools.HallOfFame(1)
        log = None
        try:
            for _ in range(n_gens):
                pop, log = algorithms.eaSimple(pop, self.toolbox, cxpb, mutpb, ngen=1, stats=self.stats,
                                               halloffame=hof, verbose=True)

                min_fitness_index = np.argmin(list(map(lambda x: x.fitness.values[0], pop)))
                pop[min_fitness_index] = hof[0]
        except KeyboardInterrupt:
            pass
        finally:
            training_stats = {
                "log": log,
            }
            print(hof[0])
            return self.agent_helper(hof[0]), training_stats
