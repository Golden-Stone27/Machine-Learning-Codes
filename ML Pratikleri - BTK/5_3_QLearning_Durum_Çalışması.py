import gymnasium as gym
import random
import numpy as np
from tqdm import tqdm

env = gym.make("Taxi-v3", render_mode = "ansi")
env.reset()
print(env.render())

'''
0: Güney
1: Kuzey
2: DOğu
3: Batı
4: yolcuyu almak
5: yolcuyu bırakmak
'''

action_space = env.action_space.n
state_space = env.observation_space.n

q_table = np.zeros((state_space, action_space))

alpha = 0.1 # Learning Rate
gamma = 0.6 # Discount Rate
epsilon = 0.1 # Epsilon

for i in tqdm(range(1, 100001)):

    state, _ = env.reset()

    done = False

    while not done:

        if random.uniform(0,1) < epsilon: # Explore %10
            action = env.action_space.sample()
        else: # Exploit
            action = np.argmax(q_table[state])

        next_state, reward, done, info, _ = env.step(action)

        q_table[state, action] = q_table[state, action] + alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action])

        state = next_state

print("Training finished")

# TEST

total_epoch, total_penalties = 0, 0
episodes = 100

for i in tqdm(range(episodes)):

    state, _ = env.reset()

    epochs, pnealties, reward = 0, 0, 0

    done = False

    while not done:

        action = np.argmax(q_table[state])

        next_state, reward, done, info, _ = env.step(action)

        state = next_state

        if reward == -10:
            pnealties += 1

        epochs +=1

    total_epoch += epochs
    total_penalties += pnealties
print("Result after {} epsiodes".format(episodes))
print("Average timesteps per episode: {}", total_epoch / episodes)
print("Average penalties per episode: {}", total_penalties / episodes)
