import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from Game_env import TowerDefenseGame

# --- High-level policy network (chooses subgoal) ---
class HighLevelPolicy(nn.Module):
    def __init__(self, obs_dim, num_subgoals, hidden_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_subgoals)
    def forward(self, obs):
        x = torch.relu(self.fc1(obs))
        x = self.fc2(x)
        return x  # logits

# --- Low-level policy network (chooses primitive action) ---
class LowLevelPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=4096):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
    def forward(self, obs_action):
        x = torch.relu(self.fc1(obs_action))
        x = self.fc2(x)
        return x

# --- HRL Agent ---
class HRLAgent:
    def __init__(self, obs_dim, action_dim, num_subgoals):
        self.high_policy = HighLevelPolicy(obs_dim, num_subgoals)
        self.low_policy = LowLevelPolicy(obs_dim, action_dim)
        self.high_optimizer = optim.Adam(self.high_policy.parameters(), lr=1e-3)
        self.low_optimizer = optim.Adam(self.low_policy.parameters(), lr=1e-3)
        self.gamma = 0.99
        self.subgoals = ["build_tower", "block_path"]  # Expand as needed

    def select_subgoal(self, obs_vec):
        obs_tensor = torch.tensor(obs_vec, dtype=torch.float32).unsqueeze(0)
        logits = self.high_policy(obs_tensor).squeeze(0)
        probs = torch.softmax(logits, dim=0)
        m = torch.distributions.Categorical(probs)
        subgoal_idx = m.sample().item()
        log_prob = m.log_prob(torch.tensor(subgoal_idx))
        return self.subgoals[subgoal_idx], log_prob

    def select_action(self, obs, valid_action_indices):
        observable_grid, waves_left = obs
        obs_flat = np.array(observable_grid, dtype=np.float32).flatten()
        obs_vec = np.concatenate([obs_flat, [waves_left]]).astype(np.float32)
        action_inputs = []
        for i, j, t in valid_action_indices:
            action_vec = np.array([i, j, t], dtype=np.float32)
            obs_action = np.concatenate([obs_vec, action_vec])
            action_inputs.append(obs_action)
        action_inputs = torch.tensor(action_inputs, dtype=torch.float32)
        logits = self.low_policy(action_inputs).squeeze(1)
        probs = torch.softmax(logits, dim=0)
        m = torch.distributions.Categorical(probs)
        idx = m.sample().item()
        log_prob = m.log_prob(torch.tensor(idx))
        return valid_action_indices[idx], log_prob

# --- Example main loop ---
def preprocess_state(env):
    grid = np.array(env.action_space)
    grid_size = grid.shape[0]
    valid_actions = []
    for i in range(grid_size):
        for j in range(grid_size):
            if grid[i, j] == 3:
                valid_actions.append([i, j, 0])
                valid_actions.append([i, j, 1])
    return np.array(valid_actions, dtype=np.int32)

def get_obs(env):
    observable_grid, waves_left = env.get_observable_space()
    obs_flat = np.array(observable_grid, dtype=np.float32).flatten()
    obs_vec = np.concatenate([obs_flat, [waves_left]]).astype(np.float32)
    return obs_vec

def main():
    env = TowerDefenseGame()
    obs_dim = env.GRID_SIZE * env.GRID_SIZE + 1
    action_dim = 3
    num_subgoals = 2  # ["build_tower", "block_path"]
    agent = HRLAgent(obs_dim, action_dim, num_subgoals)

    for episode in range(500):
        env.reset()
        obs = env.get_observable_space()
        obs_vec = get_obs(env)
        episode_reward = 0

        # High-level: choose subgoal every N steps or when needed
        subgoal, high_log_prob = agent.select_subgoal(obs_vec)

        for wave in range(500):
            if subgoal == "build_tower":
                valid_action_indices = preprocess_state(env)
                if len(valid_action_indices) == 0:
                    break
                (i, j, t), low_log_prob = agent.select_action(obs, valid_action_indices)
                if env.check_valid_action(i, j, 2):
                    try:
                        env.place_structure_index(i, j, 2, tower_type=t)
                    except:
                        continue
            # Add more subgoal logic as needed

            next_state, next_observation, reward, done, _ = env.step()
            obs = env.get_observable_space()
            obs_vec = get_obs(env)
            episode_reward += reward
            if done:
                break

        print(f"Episode {episode+1}, Reward: {episode_reward}")

if __name__ == "__main__":
    main()