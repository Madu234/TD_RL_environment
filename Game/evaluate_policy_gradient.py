import torch
import numpy as np
import os
from Game_env import TowerDefenseGame
from policy_gradient_masked import PolicyNetwork
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

def evaluate_model(model_path, state_dim, action_dim):
    env = TowerDefenseGame(training_mode=False)
    model = PolicyNetwork(state_dim, action_dim)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    env.reset()
    total_reward = 0

    for wave in range(1):  # Evaluate only one wave
        while env.number_valid_actions():
            valid_action_indices = preprocess_state(env)
            if len(valid_action_indices) == 0:
                break
            with torch.no_grad():
                logits = model.fc2(torch.relu(model.fc1(torch.zeros(1, state_dim))))
                grid_size = int(np.sqrt(state_dim))
                global_indices = [i * grid_size * 2 + j * 2 + t for i, j, t in valid_action_indices]
                valid_logits = logits[0, global_indices]
                probs = torch.softmax(valid_logits, dim=0)
                m = torch.distributions.Categorical(probs)
                idx = m.sample().item()
                i, j, type_ = valid_action_indices[idx]
            if env.check_valid_action(i, j, 2):
                try:
                    env.place_structure_index(i, j, 2, tower_type=type_)
                except:
                    continue
            else:
                continue
            next_state, next_observation, reward, done, _ = env.step()
            total_reward += reward
            if done:
                break
    return total_reward

episode_rewards = defaultdict(list)  # episode_num -> list of rewards
model_dir = os.path.join(os.path.dirname(__file__), "../PG_Masked4096_300episodes41_batch")
def main(map_name=None):
    
    if not os.path.isdir(model_dir):
        print(f"Model directory not found: {model_dir}")
        return

    env = TowerDefenseGame(training_mode=False, selected_map=map_name)
    state_dim = env.GRID_SIZE * env.GRID_SIZE
    action_dim = env.GRID_SIZE * env.GRID_SIZE * 2

    model_files = [f for f in os.listdir(model_dir) if f.endswith(".pt")]
    model_files.sort()
    print (f"Evaluating {map_name}")
    for model_file in model_files:
        # Extract the number before _1
        
        match = re.search(r'_(\d+)_1\.pt$', model_file)
        if match:
            episode_num = int(match.group(1))
        else:
            episode_num = None  # or handle as needed

        model_path = os.path.join(model_dir, model_file)
        reward = evaluate_model(model_path, state_dim, action_dim)
        #print(f"Model: {model_file} (episode {episode_num}), Reward for one wave: {reward} for map {map_name}")
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
    #print(rewards_array)

    # Print the average reward for each episode
    episode_nums = [ep for ep in sorted(episode_rewards)]
    avg_rewards = []
    for idx, rewards in enumerate(rewards_array):
        avg = np.mean(rewards) if rewards else float('nan')
        avg_rewards.append(avg)
#        print(f"Episode {episode_nums[idx]}: Average Reward = {avg}")

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
    plt.title('Average Reward per Episode (Evaluation vs Training)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

