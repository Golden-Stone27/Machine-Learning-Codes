import gymnasium as gym
import random
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

env = gym.make("FrozenLake-v1", is_slippery=False, render_mode = "ansi")
env.reset()

nb_states = env.observation_space.n
nb_actions = env.action_space.n
qtable = np.zeros((nb_states, nb_actions))

print("Q-table:")
print(qtable)

episodes = 1000
alpha = 0.5 # Learning Rate
gamma = 0.9 # Discount Rate

outcomes = []

# training
for _ in tqdm(range(episodes)):
    state, _= env.reset()
    done = False
    outcomes.append("Failure")

    while not done: #ajan başarılı olana kadar state içerisinde hareket et (action sec ve uygula)

        # action
        if np.max(qtable[state]) > 0:
            action = np.argmax(qtable[state])
        else:
            action = env.action_space.sample()

        new_state, reward, done, info, _ = env.step(action)

        # update q table
        qtable[state, action] = qtable[state, action] + alpha * (reward + gamma * np.max(qtable[new_state]) - qtable[state, action])

        state = new_state

        if reward:
            outcomes[-1] = "Success"
print("Qtable After Training: ")
print(qtable)

plt.bar(range(episodes), outcomes)
plt.show()


# TEST
epsiodes = 100
nb_success = 0


for _ in tqdm(range(episodes)):
    state, _= env.reset()
    done = False

    while not done: #ajan başarılı olana kadar state içerisinde hareket et (action sec ve uygula)

        # action
        if np.max(qtable[state]) > 0:
            action = np.argmax(qtable[state])
        else:
            action = env.action_space.sample()

        new_state, reward, done, info, _ = env.step(action)

        state = new_state

        nb_success += reward

print("Success rate: ", 100 * nb_success / episodes)

