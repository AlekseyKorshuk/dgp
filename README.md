# Decision Genetic Programming

In this project we applied genetic programming to solve OpenAI Gym Environments and compared its performance to RL
models.

# Paper

The paper with the complete evaluations, results and limitations of this project can be found [here]().

# Quick Start

## Installation

```bash
git clone git@github.com:AlekseyKorshuk/YaES.git
cd YaES
pip install -r requirements.txt
```

## Dash application

You can easily evaluate any GYM environment with our dash application. Just run the following command and open the link
in your browser.

```bash
python3 dash_app.py
```

## Demo gym environment

Evaluate PPO, MultiTree and Modi agents on the CartPole-v1 environment.

```bash
python3 evaluate.py
```

# Examples

<p float="left">
  <img src="https://user-images.githubusercontent.com/70323559/205954264-ef4c999c-1770-4277-98fb-5af888e5f0a0.gif" alt="mountain_car" height="250"/>
  <img src="https://user-images.githubusercontent.com/70323559/205955271-b68d18e5-4def-42b2-82d9-51c0fb76e853.gif" alt="cart_pole" height="250"/>
  <img src="https://user-images.githubusercontent.com/70323559/205971663-8e056a50-0044-4f7b-b7c1-dbec6ced8809.gif" alt="cart_pole" height="250"/>
</p>

# Explanations

> Why even try?

In most simple games the mapping from a state to an action can be expressed as closed-form function. It is a natural
application of genetic programming and we leverage this technique to find the exact formula.

## Single Action Space

Genetic Programming is naturally applicable here. A mathematical formula can be expressed as a tree where root is the
result of calculations, internal nodes are operations and terminal nodes are either the input variables (state of the
game in our case) or functions without variables such as constants and random number generators.

![image](https://user-images.githubusercontent.com/70323559/205684823-2c7acccd-88ed-4b20-978d-82051a9b15c9.png)

Picture source: [Wikipedia](https://upload.wikimedia.org/wikipedia/commons/7/77/Genetic_Program_Tree.png)

### Decision Making

For binary actions (do or don't do) we make a decision by checking whether the output is greater (do) or less (don't do)
than zero. For continuous actions, such as the speed of a car, we return the output as it is.

### Fitness Function

We obtain the fitness by taking the reward after running our agents in a Gym.

## Mutliple Action Space

Evolution of the usual tree doesn't scale to games with multiple outputs because it returns only single number. For that
reason, we implemented modified individuals which return vector of outputs. For discrete games we apply argmax function
and return the result as an action. In games with continuous actions we return the result unaltered.

### Modi
[Source of idea](https://www.researchgate.net/publication/228824043_A_multiple-output_program_tree_structure_in_genetic_programming)

Files with implementation:

* `agent/base.py`
* `agent/modi.py`

We implemented this idea with a slight modification. The authors of above mentioned paper suggest to add a special node
which passes the result of their calculations to the parent (as usual), but also adds this result to the output vector.
Each such node has an assigned number which specifies the index to which it will add the result.

Instead, we decided to separate these two functions. We add a special node called 'modi{index}' which passes its input
to the parent without changes and adds this input to the output vector. This approach allowed us to simplify the
implementation.


### Multi-Tree
[Source of idea](https://github.com/DEAP/deap/issues/491)

Files with implementation:

* `agent/base.py`
* `agent/multi_tree.py`

The idea is to create a bag of trees where each one is responsible for specific output index. Thus, for output vector
with size N we have N populations. To obtain an action, we take i-th individual from each population, feed them the
state of the game and collect outputs.




