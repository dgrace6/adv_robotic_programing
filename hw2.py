import numpy as np
import matplotlib.pyplot as plt

# VARIABLES
prob_moves = 0.7
prob_stays = 0.2
prob_moves_2 = 0.1

prob_door_wall = 0.3
prob_door_door = 0.8


def bel_bar(belief):
    '''
    Function that takes in an n-dim belief array and returns a belief bar array of equal size
    '''
    bel_bar = np.zeros(shape=belief.shape, dtype="float64")
    for n in range(belief.shape[0]):
        if n == 0:
            bel_bar[n] = prob_stays * belief[n]
        elif n == 1:
            bel_bar[n] = prob_moves * belief[n - 1] + prob_stays * belief[n]
        else:
            bel_bar[n] = prob_moves_2 * belief[n - 2] + prob_moves * belief[n - 1] + prob_stays * belief[n]
    return bel_bar


def bel(belief_bar):
    '''
    Function that takes in an n-dim belief bar array and returns a belief array of equal size
    '''
    bel = np.zeros(shape=belief_bar.shape, dtype="float64")
    for n in range(belief_bar.shape[0]):
        if n % 2 == 0:
            bel[n] = prob_door_wall * belief_bar[n]

        elif n % 2 == 1:
            bel[n] = prob_door_door * belief_bar[n]
    n = 1 / sum(bel)
    bel = bel * n
    return bel


belief_bar_0 = 0.25 * np.ones(4)
belief_0 = belief_bar_0

belief_bar_1 = bel_bar(belief_bar_0)
belief_1 = bel(belief_bar_1)

belief_bar_2 = bel_bar(belief_1)
belief_2 = bel(belief_bar_2)

print("Bar Belief 1: ", belief_bar_1)
print("Belief 1: ", belief_1)
print("Bar Belief 2: ", belief_bar_2)
print("Belief 2: ", belief_2)

Hallway = ['P0','P1','P2','P3']

plt.figure(1)
plt.bar(Hallway, belief_0)
plt.title('Step 1')
plt.xlabel('Belief')
plt.show()

plt.figure(2)
plt.bar(Hallway, belief_1)
plt.title('Step 2')
plt.xlabel('Belief')
plt.show()

plt.figure(3)
plt.bar(Hallway, belief_2)
plt.title('Step 3')
plt.xlabel('Belief')
plt.show()
