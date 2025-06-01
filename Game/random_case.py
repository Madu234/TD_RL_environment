import random
import os
import matplotlib.pyplot as plt
import numpy as np
from Game_env import TowerDefenseGame

def random_placement(env, num_towers):
    for _ in range(num_towers):
        while True:
            i = random.randint(0, env.GRID_SIZE - 1)
            j = random.randint(0, env.GRID_SIZE - 1)
            tower_type = random.choice([1, 2])  # Randomly choose tower type (1 or 2)
            if env.check_valid_action(i, j, 2):  # Check if the position is valid
                try:
                    env.place_structure_index(i, j, 2, tower_type=tower_type - 1)
                    break
                except Exception as e:
                    continue

def get_next_filename(base_name, extension, folder):
    i = 1
    while os.path.exists(os.path.join(folder, f"{base_name}{i}.{extension}")):
        i += 1
    return os.path.join(folder, f"{base_name}{i}.{extension}")

def main():
    env = TowerDefenseGame()
    env.reset()
    num_towers = 10  # Number of towers to place randomly
    random_placement(env, num_towers)

    rewards = []
    avg_rewards = []
    rate_episode = 10
    episodes = 1000
    title = 'RandomAgent_1000episodes'

    for e in range(episodes):
        env.reset()
        total_reward = 0
        for wave in range(500):
            random_placement(env, num_towers)
            _, _, reward, done, _ = env.step()
            total_reward += reward
            if done:
                break
        rewards.append(total_reward)

        if (e + 1) % rate_episode == 0:
            avg_reward = np.mean(rewards[-10:])
            avg_rewards.append(avg_reward)
            print(f"Episode {e + 1}, Average Reward: {avg_reward}")

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

    print("Random case results saved.")

if __name__ == "__main__":
    main()
