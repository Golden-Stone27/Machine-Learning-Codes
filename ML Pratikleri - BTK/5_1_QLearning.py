import gymnasium as gym
import random
import numpy as np

env = gym.make("FrozenLake-v1", is_slippery=False, render_mode = "ansi")
env.reset()

nb_states = env.observation_space.n
nb_actions = env.action_space.n
qtable = np.zeros((nb_states, nb_actions))

print("Q-table:")
print(qtable)

action = env.action_space.sample()
"""
sol: 0
aşağı: 1
sağ: 2
yukarı: 3
"""

# S1 --> (Action 1) --> S2
new_state, reward, done, info, _ = env.step(action)

