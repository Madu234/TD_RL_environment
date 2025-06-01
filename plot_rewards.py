import matplotlib.pyplot as plt
import numpy as np

plt.plot(episodes, rewards, color='blue', alpha=0.3, label='Average (Last 10 Episodes)')
plt.plot(episodes, avg_rewards_50, color='orange', linestyle='--', label='Average (Last 50 Episodes)')
plt.plot(episodes, trendline, color='green', linestyle='-.', label='Trendline')

plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Advanced Rewards Plot: DQN128_1001episodes6.txt')
plt.legend()
plt.grid(True)

plt.show()
