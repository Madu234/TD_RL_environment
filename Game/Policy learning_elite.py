import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import sys
import matplotlib.pyplot as plt
from Game_env import TowerDefenseGame
import os

print(sys.executable)

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=-1)
        return x

class PolicyGradientAgent:
    def __init__(self, state_dim, action_dim, episodes, gamma=0.99, lr=1e-3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.episodes = episodes
        self.log_probs = []
        self.rewards = []

        # Elite buffer setup
        self.elite_memory = deque(maxlen=5)
        self.best_reward = -float('inf')
        self.use_elite_every = 20  # Replay elite every N episodes

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        self.log_probs.append(dist.log_prob(action))
        return action.item()

    def store_reward(self, reward):
        self.rewards.append(reward)

    def finish_episode(self, episode_num):
        G = 0
        returns = []
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        # Standard loss
        loss = sum(-log_prob * R for log_prob, R in zip(self.log_probs, returns))

        # If elite episode is due and we have memory, apply it
        if episode_num % self.use_elite_every == 0 and self.elite_memory:
            for log_prob, R in self.elite_memory:
                loss += -log_prob * R  # replay elite trajectory

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        total_reward = sum(self.rewards)
        if total_reward > self.best_reward:
            self.best_reward = total_reward
            self.elite_memory = [(lp.detach(), R.detach()) for lp, R in zip(self.log_probs, returns)]

        # Clear memory
        self.log_probs = []
        self.rewards = []

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
    title = 'PG64_1000episodes_elite'
    agent = PolicyGradientAgent(state_dim, action_dim, episodes)

    rewards = []
    avg_rewards = []
    rate_episode = 10

    for e in range(episodes):
        env.reset()
        observation = preprocess_state(env)
        total_reward = 0

        for _ in range(500):
            if not env.number_valid_actions():
                break

            while True:
                action = agent.act(observation)
                i = action // (env.GRID_SIZE * 2)
                j = (action % (env.GRID_SIZE * 2)) // 2
                type = 1 if (action % 2) == 0 else 2

                if env.check_valid_action(i, j, 2):
                    try:
                        env.place_structure_index(i, j, 2, tower_type=type - 1)
                    except:
                        continue
                    break

            _, next_observation, reward, done, _ = env.step()
            next_observation = preprocess_state(env)
            agent.store_reward(reward)
            observation = next_observation
            total_reward += reward

            if done:
                print(f"episode: {e}/{episodes}, score: {total_reward}")
                break

        agent.finish_episode(e)
        rewards.append(total_reward)

        if (e + 1) % rate_episode == 0:
            avg_reward = np.mean(rewards[-10:])
            avg_rewards.append(avg_reward)
            print(f"Episode {e + 1}, Average Reward: {avg_reward}")

    # Save results
    plt.plot(avg_rewards)
    plt.xlabel(f'Average Reward per {rate_episode} Episodes')
    plt.ylabel('Average Reward')
    plt.title(title)

    if not os.path.exists(title):
        os.makedirs(title)

    png_filename = get_next_filename(title, 'png', title)
    plt.savefig(png_filename)

    txt_filename = get_next_filename(title, 'txt', title)
    with open(txt_filename, 'w') as f:
        for reward in rewards:
            f.write(f"{reward:.2f}\n")

    average_title = f"average_{title}"
    txt_filename2 = get_next_filename(average_title, 'txt', title)
    with open(txt_filename2, 'w') as f:
        for avg_reward in avg_rewards:
            f.write(f"{avg_reward:.2f}\n")

    model_filename = get_next_filename(title, 'pt', title)
    torch.save(agent.policy.state_dict(), model_filename)

    plt.clf()

if __name__ == "__main__":
    for _ in range(5):
        main()
