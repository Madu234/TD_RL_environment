import matplotlib.pyplot as plt
import numpy as np

# Load reward data
with open("DQN128_1000episodes9.txt", "r") as f:
    rewards = [float(line.strip()) for line in f]

# Moving average
def moving_average(data, window_size=50):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Compute smoothed and cumulative averages
window_size = 50
smoothed_rewards = moving_average(rewards, window_size)
cumulative_avg = np.cumsum(rewards) / (np.arange(len(rewards)) + 1)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(rewards, alpha=0.3, label="Raw Reward")
plt.plot(np.arange(window_size - 1, len(rewards)), smoothed_rewards, label=f"Moving Avg (window={window_size})", linewidth=2)
plt.plot(cumulative_avg, label="Cumulative Average", linestyle='--', color='black', linewidth=2)

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Reward Trends Over Episodes")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("reward_trends_with_cumulative.png")
plt.show()
