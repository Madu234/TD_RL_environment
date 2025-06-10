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
episodes = 2000
neurons = 4096
title = f"SARSA{neurons}_{episodes}episodes{seed}_batch"
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=4096):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)  # Output Q-value

    def forward(self, obs_action):
        x = torch.relu(self.fc1(obs_action))
        x = self.fc2(x)
        return x  # shape: (batch, 1)

class SARSAAgent:
    def __init__(self, obs_dim, action_dim, lr=0.01, gamma=0.99, epsilon=0.1):
        self.q_net = QNetwork(obs_dim, action_dim)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon

    def select_action(self, obs, valid_action_indices):
        observable_grid, waves_left = obs
        obs_flat = np.array(observable_grid, dtype=np.float32).flatten()
        obs_vec = np.concatenate([obs_flat, [waves_left]]).astype(np.float32)

        # Prepare action representations
        action_inputs = []
        for i, j, t in valid_action_indices:
            action_vec = np.array([i, j, t], dtype=np.float32)
            obs_action = np.concatenate([obs_vec, action_vec])
            action_inputs.append(obs_action)
        action_inputs = torch.tensor(action_inputs, dtype=torch.float32)  # shape: (num_valid, obs_dim+action_dim)

        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            idx = random.randint(0, len(valid_action_indices) - 1)
            return valid_action_indices[idx], idx

        with torch.no_grad():
            q_values = self.q_net(action_inputs).squeeze(1)  # shape: (num_valid,)
            idx = torch.argmax(q_values).item()
        return valid_action_indices[idx], idx

    def update(self, obs, action, reward, next_obs, next_valid_action_indices, next_action_idx, done):
        observable_grid, waves_left = obs
        obs_flat = np.array(observable_grid, dtype=np.float32).flatten()
        obs_vec = np.concatenate([obs_flat, [waves_left]]).astype(np.float32)
        action_vec = np.array(action, dtype=np.float32)
        obs_action = np.concatenate([obs_vec, action_vec])
        obs_action_tensor = torch.tensor(obs_action, dtype=torch.float32).unsqueeze(0)  # (1, obs_dim+action_dim)

        q_pred = self.q_net(obs_action_tensor).squeeze(0)  # (1,)

        # Compute target
        if done or len(next_valid_action_indices) == 0:
            q_target = torch.tensor(reward, dtype=torch.float32)
        else:
            next_observable_grid, next_waves_left = next_obs
            next_obs_flat = np.array(next_observable_grid, dtype=np.float32).flatten()
            next_obs_vec = np.concatenate([next_obs_flat, [next_waves_left]]).astype(np.float32)
            next_action = next_valid_action_indices[next_action_idx]
            next_action_vec = np.array(next_action, dtype=np.float32)
            next_obs_action = np.concatenate([next_obs_vec, next_action_vec])
            next_obs_action_tensor = torch.tensor(next_obs_action, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_next = self.q_net(next_obs_action_tensor).squeeze(0)
            q_target = reward + self.gamma * q_next

        loss = nn.MSELoss()(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def preprocess_state(env):
    grid = np.array(env.action_space)
    grid_size = grid.shape[0]
    valid_actions = []
    for i in range(grid_size):
        for j in range(grid_size):
            if grid[i, j] == 3:
                idx1 = [i, j, 0]
                idx2 = [i, j, 1]
                valid_actions.append(idx1)
                valid_actions.append(idx2)
    return np.array(valid_actions, dtype=np.int32)

def get_next_filename(base_name, extension, folder):
    i = 1
    while os.path.exists(os.path.join(folder, f"{base_name}{i}.{extension}")):
        i += 1
    return os.path.join(folder, f"{base_name}{i}.{extension}")

def adjust_learning_rate(agent, avg_reward, target=29.0, base_lr=0.01, min_lr=1e-4, max_lr=0.1):
    distance = abs(target - avg_reward)
    norm_dist = min(distance / target, 1.0)
    new_lr = min_lr + (max_lr - min_lr) * norm_dist
    for param_group in agent.optimizer.param_groups:
        param_group['lr'] = new_lr
    return new_lr

def main():
    env = TowerDefenseGame()
    obs_dim = env.GRID_SIZE * env.GRID_SIZE + 1  # +1 for waves_left
    action_dim = 3  # [i, j, t]
    agent = SARSAAgent(obs_dim, action_dim)

    rewards = []
    avg_rewards = []
    rate_episode = 10
    last_avg = None
    stuck_count = 0
    batch_size = 5
    episode_counter = 0

    for e in range(episodes):
        env.reset()
        episode_reward = 0
        obs = env.get_observable_space()
        valid_action_indices = preprocess_state(env)
        action, action_idx = agent.select_action(obs, valid_action_indices)

        for wave in range(500):
            # Take action
            if env.check_valid_action(action[0],action[1], 2):
                try:
                    env.place_structure_index(action[0], action[1], 2, tower_type=action[2])
                except:
                    pass

            next_state, next_observation, reward, done, _ = env.step()
            next_obs = env.get_observable_space()
            next_valid_action_indices = preprocess_state(env)
            if len(next_valid_action_indices) == 0:
                break  # No valid actions, end the episode

            next_action, next_action_idx = agent.select_action(next_obs, next_valid_action_indices)

            agent.update(obs, action, reward, next_obs, next_valid_action_indices, next_action_idx, done)

            obs = next_obs
            action = next_action
            action_idx = next_action_idx
            episode_reward += reward
            if done:
                break

        episode_counter += 1

        rewards.append(episode_reward)

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

        if (e + 1) % rate_episode == 0:
            avg_reward = np.mean(rewards[-10:])
            avg_rewards.append(avg_reward)
            lr = adjust_learning_rate(agent, avg_reward)
            print(f"Episode {e + 1}, Average Reward: {avg_reward}, Learning Rate: {lr:.5f}")

            # Stuck detection and restart
            if last_avg is not None and avg_reward == last_avg:
                stuck_count += 1
            else:
                stuck_count = 0
            last_avg = avg_reward
            if stuck_count >= 5:
                print("Average reward stuck for 5 consecutive checks. Saving stuck model and restarting script...")
                stuck_n = 1
                if not os.path.exists(title):
                    os.makedirs(title)
                while True:
                    stuck_model_path = os.path.join(title, f"stuck_model_{title}_{stuck_n}.pt")
                    if not os.path.exists(stuck_model_path):
                        torch.save(agent.q_net.state_dict(), stuck_model_path)
                        break
                    stuck_n += 1
                os.execv(sys.executable, [sys.executable] + sys.argv)

    # Plot the average rewards
    plt.plot(avg_rewards)
    plt.xlabel(f'Average Reward per {rate_episode} Episodes')
    plt.ylabel('Average Reward')
    plt.title(f'{title}')

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

    plt.clf()

    subprocess.run(['python3', '/home/madu/Desktop/TD_RL_environment/Game/advance_plot.py', title])

if __name__ == "__main__":
    main()