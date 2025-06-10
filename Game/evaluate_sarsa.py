import torch
import numpy as np
import os
from Game_env import TowerDefenseGame
from SARSA import QNetwork
import re
from collections import defaultdict
import matplotlib.pyplot as plt

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

def evaluate_model(model_path, obs_dim, action_dim):
    env = TowerDefenseGame(training_mode=False)
    model = QNetwork(obs_dim, action_dim)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    env.reset()
    total_reward = 0

    obs = env.get_observable_space()
    for wave in range(500):
        valid_action_indices = preprocess_state(env)
        if len(valid_action_indices) == 0:
            break
        # Prepare input for all valid actions
        observable_grid, waves_left = obs
        obs_flat = np.array(observable_grid, dtype=np.float32).flatten()
        obs_vec = np.concatenate([obs_flat, [waves_left]]).astype(np.float32)
        action_inputs = []
        for i, j, t in valid_action_indices:
            action_vec = np.array([i, j, t], dtype=np.float32)
            obs_action = np.concatenate([obs_vec, action_vec])
            action_inputs.append(obs_action)
        action_inputs = torch.tensor(action_inputs, dtype=torch.float32)  # shape: (num_valid, obs_dim+action_dim)
        with torch.no_grad():
            q_values = model(action_inputs).squeeze(1)  # shape: (num_valid,)
            idx = torch.argmax(q_values).item()
            i, j, t = valid_action_indices[idx]
        if env.check_valid_action(i, j, 2):
            try:
                env.place_structure_index(i, j, 2, tower_type=t)
            except:
                continue
        next_state, next_observation, reward, done, _ = env.step()
        obs = env.get_observable_space()
        total_reward += reward
        if done:
            break
    return total_reward

episode_rewards = defaultdict(list)
model_dir = os.path.join(os.path.dirname(__file__), "../SARSA4096_2000episodes40_batch")  # Adjust as needed

def main(map_name=None):
    if not os.path.isdir(model_dir):
        print(f"Model directory not found: {model_dir}")
        return

    env = TowerDefenseGame(training_mode=False, selected_map=map_name)
    obs_dim = env.GRID_SIZE * env.GRID_SIZE + 1
    action_dim = 3

    model_files = [f for f in os.listdir(model_dir) if f.endswith(".pt")]
    model_files.sort()
    print(f"Evaluating {map_name}")
    for model_file in model_files:
        match = re.search(r'_(\d+)_1\.pt$', model_file)
        if match:
            episode_num = int(match.group(1))
        else:
            episode_num = None

        model_path = os.path.join(model_dir, model_file)
        reward = evaluate_model(model_path, obs_dim, action_dim)
        if episode_num is not None:
            episode_rewards[episode_num].append(reward)

if __name__ == "__main__":
    folder = os.path.dirname(__file__)
    folder = os.path.join(folder, "Maps", 'Evaluation')
    files = []
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            files.append(file)
            main(file)
    # Convert to array of arrays (list of lists), sorted by episode_num
    rewards_array = [episode_rewards[ep] for ep in sorted(episode_rewards)]

    # Print the average reward for each episode
    episode_nums = [ep for ep in sorted(episode_rewards)]
    avg_rewards = []
    for idx, rewards in enumerate(rewards_array):
        avg = np.mean(rewards) if rewards else float('nan')
        avg_rewards.append(avg)

    # Plot average reward per episode (evaluation) using moving average
    plt.figure(figsize=(10, 6))
    window = 10
    if len(avg_rewards) >= window:
        eval_ma = np.convolve(avg_rewards, np.ones(window)/window, mode='valid')
        eval_ma_episodes = episode_nums[window-1:]
        plt.plot(eval_ma_episodes, eval_ma, label=f'Evaluation Moving Avg ({window})', marker='o')
    else:
        plt.plot(episode_nums, avg_rewards, label='Evaluation Average Reward', marker='o')

    # Trend line for evaluation (on moving average if available)
    if len(avg_rewards) >= window:
        z = np.polyfit(eval_ma_episodes, eval_ma, 1)
        p = np.poly1d(z)
        plt.plot(eval_ma_episodes, p(eval_ma_episodes), "r--", label='Evaluation Trend Line')
    elif len(episode_nums) > 1:
        z = np.polyfit(episode_nums, avg_rewards, 1)
        p = np.poly1d(z)
        plt.plot(episode_nums, p(episode_nums), "r--", label='Evaluation Trend Line')

    plt.xlabel('Episode')
    plt.ylabel('Average evaluation Reward')
    plt.title('SARSA: Average Reward per Episode (Evaluation)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()