import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
from Game_env import TowerDefenseGame
import os
import subprocess
import sys

seed = 40
episodes = 500
neurons = 4096
title = f"DQN{neurons}_{episodes}episodes{seed}_batch"
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

class DQNNetwork(nn.Module):
    def __init__(self, obs_dim, num_actions, hidden_dim=4096):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_actions)

    def forward(self, obs):
        x = torch.relu(self.fc1(obs))
        x = self.fc2(x)
        return x  # shape: (batch, num_actions)

class DQNAgent:
    def __init__(self, obs_dim, num_actions, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995, buffer_size=10000, batch_size=64):
        self.q_net = DQNNetwork(obs_dim, num_actions)
        self.target_net = DQNNetwork(obs_dim, num_actions)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.update_target_steps = 1000
        self.learn_step = 0

    def select_action(self, obs, valid_action_mask):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        if random.random() < self.epsilon:
            valid_indices = np.where(valid_action_mask)[0]
            idx = np.random.choice(valid_indices)
            return idx
        with torch.no_grad():
            q_values = self.q_net(obs_tensor).squeeze(0).cpu().numpy()
            q_values[~valid_action_mask] = -np.inf  # Mask invalid actions
            idx = int(np.argmax(q_values))
        return idx

    def store(self, state, action, reward, next_state, done, next_valid_mask):
        self.memory.append((state, action, reward, next_state, done, next_valid_mask))

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones, next_valid_masks = zip(*batch)
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        next_valid_masks = np.array(next_valid_masks)

        q_values = self.q_net(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_net(next_states).cpu().numpy()
            # Mask invalid actions in the next state
            next_q_values[~next_valid_masks] = -np.inf
            next_q_value = torch.tensor(np.max(next_q_values, axis=1), dtype=torch.float32)

        q_target = rewards + self.gamma * next_q_value * (1 - dones)
        loss = nn.MSELoss()(q_value, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step += 1
        if self.learn_step % self.update_target_steps == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

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
    num_actions = grid_size * grid_size * 2  # [i, j, t] flattened
    agent = DQNAgent(obs_dim, num_actions)

    rewards = []
    avg_rewards = []
    rate_episode = 10

    for e in range(episodes):
        env.reset()
        obs_vec = get_obs(env)
        episode_reward = 0

        for wave in range(500):
            valid_action_mask = preprocess_state(env)
            if not valid_action_mask.any():
                break
            action_idx = agent.select_action(obs_vec, valid_action_mask)
            i, j, t = action_index_to_tuple(action_idx, grid_size)
            if env.check_valid_action(i, j, 2):
                try:
                    env.place_structure_index(i, j, 2, tower_type=t)
                except:
                    pass

            next_state, next_observation, reward, done, _ = env.step()
            next_obs_vec = get_obs(env)
            next_valid_action_mask = preprocess_state(env)
            agent.store(obs_vec, action_idx, reward, next_obs_vec, done, next_valid_action_mask)
            agent.update()
            obs_vec = next_obs_vec
            episode_reward += reward
            if done:
                break

        agent.decay_epsilon()
        rewards.append(episode_reward)

        if (e + 1) % rate_episode == 0:
            avg_reward = np.mean(rewards[-rate_episode:])
            avg_rewards.append(avg_reward)
            print(f"Episode {e + 1}, Average Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

        # Save model every 10 episodes
        if (e + 1) % 10 == 0:
            save_idx = 1
            if not os.path.exists(title):
                os.makedirs(title)
            while True:
                model_save_path = os.path.join(title, f"model_{title}_{e+1}_{save_idx}.pt")
                if not os.path.exists(model_save_path):
                    torch.save(agent.q_net.state_dict(), model_save_path)
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