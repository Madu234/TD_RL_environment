import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
from Game_env import TowerDefenseGame
import os
import subprocess
import sys

seed = 40
episodes = 300
neurons = 4096
title = f"AC{neurons}_{episodes}episodes{seed}_batch"
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

class Actor(nn.Module):
    def __init__(self, obs_dim, num_actions, hidden_dim=4096):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_actions)
    def forward(self, obs):
        x = torch.relu(self.fc1(obs))
        x = self.fc2(x)
        return x  # logits

class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_dim=4096):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
    def forward(self, obs):
        x = torch.relu(self.fc1(obs))
        x = self.fc2(x)
        return x  # value

class ACAgent:
    def __init__(self, obs_dim, num_actions, actor_lr=1e-4, critic_lr=1e-3, gamma=0.99):
        self.actor = Actor(obs_dim, num_actions)
        self.critic = Critic(obs_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.num_actions = num_actions

    def select_action(self, obs, valid_action_mask):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits = self.actor(obs_tensor).squeeze(0)
        logits[~valid_action_mask] = -float('inf')
        probs = torch.softmax(logits, dim=0)
        m = torch.distributions.Categorical(probs)
        action_idx = m.sample().item()
        log_prob = m.log_prob(torch.tensor(action_idx))
        return action_idx, log_prob

    def compute_returns(self, rewards, dones, next_value):
        returns = []
        R = next_value
        for r, done in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1. - done)
            returns.insert(0, R)
        return returns

    def update(self, trajectories):
        states, actions, log_probs, rewards, dones, next_states, masks = zip(*trajectories)
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        log_probs = torch.stack(log_probs)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)

        with torch.no_grad():
            next_values = self.critic(next_states).squeeze(1)
        returns = self.compute_returns(rewards.tolist(), dones.tolist(), next_values[-1].item())
        returns = torch.tensor(returns, dtype=torch.float32)

        values = self.critic(states).squeeze(1)
        advantage = returns - values

        # Actor loss
        actor_loss = -(log_probs * advantage.detach()).mean()
        # Critic loss
        critic_loss = advantage.pow(2).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

def preprocess_state(env):
    grid = np.array(env.action_space)
    grid_size = grid.shape[0]
    valid_mask = np.zeros(grid_size * grid_size * 2, dtype=bool)
    for i in range(grid_size):
        for j in range(grid_size):
            if grid[i, j] == 3:
                idx1 = i * grid_size * 2 + j * 2 + 0
                idx2 = i * grid_size * 2 + j * 2 + 1
                valid_mask[idx1] = True
                valid_mask[idx2] = True
    return valid_mask

def action_index_to_tuple(idx, grid_size):
    i = idx // (grid_size * 2)
    j = (idx % (grid_size * 2)) // 2
    t = idx % 2
    return i, j, t

def get_obs(env):
    observable_grid, waves_left = env.get_observable_space()
    obs_flat = np.array(observable_grid, dtype=np.float32).flatten()
    obs_vec = np.concatenate([obs_flat, [waves_left]]).astype(np.float32)
    return obs_vec

def main():
    env = TowerDefenseGame()
    grid_size = env.GRID_SIZE
    obs_dim = grid_size * grid_size + 1
    num_actions = grid_size * grid_size * 2
    agent = ACAgent(obs_dim, num_actions)

    rewards = []
    avg_rewards = []
    rate_episode = 10

    for e in range(episodes):
        env.reset()
        obs_vec = get_obs(env)
        episode_reward = 0
        trajectories = []

        for wave in range(500):
            valid_action_mask = preprocess_state(env)
            if not valid_action_mask.any():
                break
            action_idx, log_prob = agent.select_action(obs_vec, valid_action_mask)
            i, j, t = action_index_to_tuple(action_idx, grid_size)
            if env.check_valid_action(i, j, 2):
                try:
                    env.place_structure_index(i, j, 2, tower_type=t)
                except:
                    pass

            next_state, next_observation, reward, done, _ = env.step()
            next_obs_vec = get_obs(env)
            next_valid_action_mask = preprocess_state(env)
            trajectories.append((obs_vec, action_idx, log_prob, reward, done, next_obs_vec, valid_action_mask))
            obs_vec = next_obs_vec
            episode_reward += reward
            if done:
                break

        agent.update(trajectories)
        rewards.append(episode_reward)

        if (e + 1) % rate_episode == 0:
            avg_reward = np.mean(rewards[-rate_episode:])
            avg_rewards.append(avg_reward)
            print(f"Episode {e + 1}, Average Reward: {avg_reward:.2f}")

        # Save model every 10 episodes
        if (e + 1) % 10 == 0:
            save_idx = 1
            if not os.path.exists(title):
                os.makedirs(title)
            while True:
                model_save_path = os.path.join(title, f"model_{title}_{e+1}_{save_idx}.pt")
                if not os.path.exists(model_save_path):
                    torch.save(agent.actor.state_dict(), model_save_path)
                    break
                save_idx += 1

    # Plot the average rewards
    plt.plot(avg_rewards)
    plt.xlabel(f'Average Reward per {rate_episode} Episodes')
    plt.ylabel('Average Reward')
    plt.title(f'{title}')

    if not os.path.exists(title):
        os.makedirs(title)

    png_filename = os.path.join(title, f"{title}_plot.png")
    plt.savefig(png_filename)

    txt_filename = os.path.join(title, f"{title}_rewards.txt")
    with open(txt_filename, 'w') as f:
        for reward in rewards:
            f.write(f"{reward:.2f}\n")

    plt.clf()

    subprocess.run(['python3', '/home/madu/Desktop/TD_RL_environment/Game/advance_plot.py', title])

if __name__ == "__main__":
    main()