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
episodes = 500
neurons = 4096
title = f"PG_Masked{neurons}_{episodes}episodes{seed}_batch"
#title = 'PG_Masked1024_1500episodes42_batch'
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=4096):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)  # Output a single logit

    def forward(self, obs_action):
        x = torch.relu(self.fc1(obs_action))
        x = self.fc2(x)
        return x  # shape: (batch, 1)

class PolicyGradientAgent:
    def __init__(self, obs_dim, action_dim, lr=0.01, gamma=0.99, epsilon=0):
        self.policy = PolicyNetwork(obs_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.log_probs = []
        self.rewards = []
        self.epsilon = epsilon

    def select_action(self, obs, valid_action_indices):
        observable_grid, waves_left = obs
        obs_flat = np.array(observable_grid, dtype=np.float32).flatten()
        obs_vec = np.concatenate([obs_flat, [waves_left]]).astype(np.float32)

        # Prepare action representations (e.g., one-hot or just [i, j, t])
        action_inputs = []
        for i, j, t in valid_action_indices:
            action_vec = np.array([i, j, t], dtype=np.float32)
            obs_action = np.concatenate([obs_vec, action_vec])
            action_inputs.append(obs_action)
        action_inputs = torch.tensor(action_inputs, dtype=torch.float32)  # shape: (num_valid, obs_dim+action_dim)

        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            idx = random.randint(0, len(valid_action_indices) - 1)
            self.log_probs.append(None)
            return valid_action_indices[idx]

        logits = self.policy(action_inputs).squeeze(1)  # shape: (num_valid,)
        probs = torch.softmax(logits, dim=0)
        m = torch.distributions.Categorical(probs)
        idx = m.sample().item()
        self.log_probs.append(m.log_prob(torch.tensor(idx)))
        return valid_action_indices[idx]

    def store_reward(self, reward):
        self.rewards.append(reward)

    def finish_episode(self):
        R = 0
        returns = []
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        loss = []
        for log_prob, R in zip(self.log_probs, returns):
            if log_prob is not None:  # Only use log_probs from policy actions
                loss.append(-log_prob * R)
        if loss:  # Only backward if there is something to optimize
            loss = torch.stack(loss).sum()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.log_probs = []
        self.rewards = []

def preprocess_state(env):
    # Return a list of valid action indices (Index-Over-Valid-Actions)
    grid = np.array(env.action_space)
    grid_size = grid.shape[0]
    valid_actions = []
    for i in range(grid_size):
        for j in range(grid_size):
            if grid[i, j] == 3:
                idx1 = [i,j,0]    # type=0
                idx2 = [i,j,1] # type=1
                valid_actions.append(idx1)
                valid_actions.append(idx2)
    return np.array(valid_actions, dtype=np.int32)

def get_next_filename(base_name, extension, folder):
    i = 1
    while os.path.exists(os.path.join(folder, f"{base_name}{i}.{extension}")):
        i += 1
    return os.path.join(folder, f"{base_name}{i}.{extension}")

def adjust_learning_rate(agent, avg_reward, target=29.0, base_lr=0.01, min_lr=1e-4, max_lr=0.1):
    # Increase lr if far from target, decrease if close
    distance = abs(target - avg_reward)
    # Normalize distance to [0, 1] (assuming max possible distance is 29)
    norm_dist = min(distance / target, 1.0)
    # Inverse: closer to target, smaller lr
    new_lr = min_lr + (max_lr - min_lr) * norm_dist
    for param_group in agent.optimizer.param_groups:
        param_group['lr'] = new_lr
    return new_lr

def main():
    env = TowerDefenseGame()
    obs_dim = env.GRID_SIZE * env.GRID_SIZE + 1  # +1 for waves_left
    #valid_action_dim = env.GRID_SIZE * env.GRID_SIZE * 2  # Two types of actions (type=0 and type=1)
    action_dim = 3  # [i, j, t]
    agent = PolicyGradientAgent(obs_dim, action_dim)

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
        actions = []
        obs = env.get_observable_space()
        for wave in range(500):
            
            while env.number_valid_actions():
                for attempt in range(100):  # Prevent infinite loop
                    valid_action_indices = preprocess_state(env)
                    # TODO: provide more context to the agend about the current state
                    i, j, type_ = agent.select_action(obs, valid_action_indices)
                    if env.check_valid_action(i, j, 2):
                        try:
                            env.place_structure_index(i, j, 2, tower_type=type_)
                            actions.append((i, j, type_))
                        except:
                            continue
                        break
                else:
                    break  # If no valid action found after 100 attempts, break

            next_state, next_observation, reward, done, _ = env.step()
#           # get observable space after the step update
            obs = env.get_observable_space()
            episode_reward += reward
            agent.store_reward(reward)
            if done:
                break

        episode_counter += 1

        if episode_counter % batch_size == 0:
            agent.finish_episode()  # Update policy using all collected episodes

        rewards.append(episode_reward)

        # Save model every 10 episodes
        if (e + 1) % 10 == 0:
            save_idx = 1
            if not os.path.exists(title):
                os.makedirs(title)
            while True:
                model_save_path = os.path.join(title, f"model_{title}_{e+1}_{save_idx}.pt")
                if not os.path.exists(model_save_path):
                    torch.save(agent.policy.state_dict(), model_save_path)
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
                        torch.save(agent.policy.state_dict(), stuck_model_path)
                        break
                    stuck_n += 1
                os.execv(sys.executable, [sys.executable] + sys.argv)

    # Plot the average rewards
    plt.plot(avg_rewards)
    plt.xlabel(f'Average Reward per {rate_episode} Episodes')
    plt.ylabel('Average Reward')
    plt.title(f'{title}')

    # Create a folder with the title name
    if not os.path.exists(title):
        os.makedirs(title)

    # Save the plot
    png_filename = get_next_filename(title, 'png', title)
    plt.savefig(png_filename)

    # Save the rewards vector
    txt_filename = get_next_filename(title, 'txt', title)
    with open(txt_filename, 'w') as f:
        for reward in rewards:
            f.write(f"{reward:.2f}\n")
    average_title = f"average_{title}"
    txt_filename2 = get_next_filename(average_title, 'txt', title)
    with open(txt_filename2, 'w') as f:
        for avg_reward in avg_rewards:
            f.write(f"{avg_reward:.2f}\n")

    plt.clf()  # Clear the current figure

    # Call advance_plot.py to process the folder
    subprocess.run(['python3', '/home/madu/Desktop/TD_RL_environment/Game/advance_plot.py', title])

if __name__ == "__main__":
    main()
