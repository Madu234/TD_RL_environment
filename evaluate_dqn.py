import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Game"))

import torch
import numpy as np
import random
from Game.Game_env import TowerDefenseGame
from Game.DQN import SimpleDQN

def preprocess_state(env):
    observation = np.array(env.action_space)
    return observation.reshape(-1)

def evaluate():
    model_path = "/home/madu/Desktop/TD_RL_environment/DQN64x2_1000episodes/best_model.pt"
    env = TowerDefenseGame(training_mode=False)
    grid_size = env.GRID_SIZE
    state_dim = grid_size * grid_size
    model = SimpleDQN(state_dim)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    env.reset()
    observation = preprocess_state(env)
    wave_reward = 0

    # Evaluate only one wave
    while env.number_valid_actions():
        penalty = 0
        tries = 1000
        noise_input = 0.0
        for _ in range(tries):
            obs_flat = np.array(observation).flatten()
            input_obs = np.concatenate([obs_flat, [noise_input]])
            state = torch.FloatTensor(input_obs).view(1, -1)
            with torch.no_grad():
                out = model(state).squeeze(0)
            type_ = int(out[0].item() >= 0.5)
            x = int(torch.round(out[1] * (grid_size - 1)).item())
            y = int(torch.round(out[2] * (grid_size - 1)).item())
            if env.check_valid_action(x, y, 2):
                try:
                    env.place_structure_index(x, y, 2, tower_type=type_)
                except:
                    #penalty += 1
                    noise_input += 1/tries
                    continue
                break
            else:
                #penalty += 1
                noise_input += 1/tries
        else:
            # If all tries fail, penalize and do random valid action
            penalty += 1
            valid_indices = np.argwhere(np.array(env.action_space) == 3)
            if len(valid_indices) > 0:
                rand_x, rand_y = random.choice(valid_indices)
                rand_type = random.randint(0, 1)
                try:
                    env.place_structure_index(rand_x, rand_y, 2, tower_type=rand_type)
                except:
                    pass

        next_state, next_observation, reward, done, _ = env.step()
        wave_reward += reward - penalty
        if done:
            break
        observation = preprocess_state(env)

    print(f"Reward for one wave: {wave_reward}")

if __name__ == "__main__":
    evaluate()
