# YaES
The aim of this project is to apply genetic programming to solve OpenAI Gym Environments and to compare its performance to RL models.

In most simple games the mapping from a state to an action can be expressed as closed-form function. It is a natural application of genetic programming and we leverage this technique to find the exact solution.

## Single Action Games
Genetic Programming is naturally applicable here. A mathematical formula can be expressed as a tree where root is the result of calculations, internal nodes are operations and terminal nodes are either the input variables (state of the game in our case) or functions without variables such as constants and random number generators.

![image](https://user-images.githubusercontent.com/70323559/205684823-2c7acccd-88ed-4b20-978d-82051a9b15c9.png)

Picture source: [Wikipedia](https://upload.wikimedia.org/wikipedia/commons/7/77/Genetic_Program_Tree.png)

### Decision Making
For binary actions (do or don't do) we make a decision by checking whether the output is greater (do) or less (don't do) than zero. For continuous actions, such as the speed of a car, we return the output as it is.

### Fitness Function
We obtain the fitness by taking the reward after running our agents in a Gym.

## Mutliple Action Games
Evolution of the usual tree doesn't scale to games with multiple outputs because it returns only single number. For that reason, we implemented several modifications.

### Modi
[Source of idea](https://www.researchgate.net/publication/228824043_A_multiple-output_program_tree_structure_in_genetic_programming)

Files with implementation:
* `agent/base.py`
* `agent/modi.py`

We implemented this idea with a slight modification. The authors of above mentioned paper suggest to add a special node which passes the result of their calculations to the parent (as usual), but also adds this result to the output vector. Each such node has an assigned number which specifies the index to which it will add the result. 

Instead, we decided to separate these two functions. We add a special node called 'modi{index}' which passes its input to the parent without changes and adds this input to the output vector. This approach allowed us to simplify the implementation.

### Multi-Tree
[Source of idea](https://github.com/DEAP/deap/issues/491)

Files with implementation:
* `agent/base.py`
* `agent/multi_tree.py`
