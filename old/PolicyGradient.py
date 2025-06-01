import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
from Game_env import TowerDefenseGame
import os

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return torch.softmax(x, dim=-1)

class PolicyGradientAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.99):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.gamma = gamma

        # Stores (log_prob, reward) tuples for the episode
        self.episode_log_probs = []
        self.episode_rewards = []

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        self.episode_log_probs.append(dist.log_prob(action))
        return action.item()

    def remember(self, reward):
        self.episode_rewards.append(reward)

    def finish_episode(self):
        R = 0
        returns = []

        # Compute discounted returns
        for r in reversed(self.episode_rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        # Normalize returns for stability
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = 0
        for log_prob, ret in zip(self.episode_log_probs, returns):
            loss -= log_prob * ret  # Gradient ascent

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Clear memory
        self.episode_rewards = []
        self.episode_log_probs = []

def preprocess_state(env):
    return np.array(env.action_space).flatten()

def get_next_filename(base_name, extension, folder):
    i = 1
    while os.path.exists(os.path.join(folder, f"{base_name}{i}.{extension}")):
        i += 1
    return os.path.join(folder, f"{base_name}{i}.{extension}")

def main():
    env = TowerDefenseGame()
    state_dim = env.GRID_SIZE * env.GRID_SIZE
    action_dim = env.GRID_SIZE * env.GRID_SIZE * 2
    episodes = 1000
    title = 'PolicyGradient128_1000episodes'
    agent = PolicyGradientAgent(state_dim, action_dim)

    rewards = []
    avg_rewards = []
    rate_episode = 10

    for e in range(episodes):
        env.reset()
        observation = preprocess_state(env)
        total_reward = 0
        actions = []

        for wave in range(500):
            if not env.number_valid_actions():
                break

            while True:
                action = agent.act(observation)
                i = action // (env.GRID_SIZE * 2)
                j = (action % (env.GRID_SIZE * 2)) // 2
                type = 1 if (action % 2) == 0 else 2

                if env.check_valid_action(i, j, 2):
                    try:
                        env.place_structure_index(i, j, 2, tower_type=type-1)
                        actions.append(action)
                    except:
                        continue
                    break
                else:
                    continue

        next_state, next_observation, reward, done, _ = env.step()
        agent.remember(reward)
        total_reward += reward

        agent.finish_episode()
        rewards.append(total_reward)

        if (e + 1) % rate_episode == 0:
            avg_reward = np.mean(rewards[-rate_episode:])
            avg_rewards.append(avg_reward)
            print(f"Episode {e+1}, Average Reward: {avg_reward:.2f}")

    # Save results
    if not os.path.exists(title):
        os.makedirs(title)

    plt.plot(avg_rewards)
    plt.xlabel(f'Average Reward per {rate_episode} Episodes')
    plt.ylabel('Average Reward')
    plt.title(title)

    png_filename = get_next_filename(title, 'png', title)
    plt.savefig(png_filename)

    txt_filename = get_next_filename(title, 'txt', title)
    with open(txt_filename, 'w') as f:
        for reward in rewards:
            f.write(f"{reward:.2f}\n")

    txt_filename2 = get_next_filename(f"average_{title}", 'txt', title)
    with open(txt_filename2, 'w') as f:
        for avg_reward in avg_rewards:
            f.write(f"{avg_reward:.2f}\n")

    model_filename = get_next_filename(title, 'pt', title)
    torch.save(agent.policy.state_dict(), model_filename)

    plt.clf()

if __name__ == "__main__":
    main()
