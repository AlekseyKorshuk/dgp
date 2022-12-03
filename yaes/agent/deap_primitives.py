from typing import Any
from deap.gp import PrimitiveSetTyped
import operator
import random


def if_then_else(cond: bool, output1: Any, output2: Any):
    return output1 if cond else output2


def safe_div(x1: float, x2: float, eps=1e-15):
    return 0 if x2 < eps else x1 / x2


def basic_primitive_set(num_inputs, name):
    pset = PrimitiveSetTyped(name, [float] * num_inputs, float)
    pset.addPrimitive(operator.xor, [bool, bool], bool)
    pset.addPrimitive(operator.mul, [float, float], float)
    pset.addPrimitive(safe_div, [float, float], float)
    pset.addPrimitive(operator.add, [float, float], float)
    pset.addPrimitive(operator.sub, [float, float], float)

    pset.addPrimitive(if_then_else, [bool, float, float], float)
    pset.addTerminal(1, bool)
    pset.addEphemeralConstant('random_number_' + name, lambda: random.uniform(-1, 1), float)

    return pset
