import matplotlib.pyplot as plt
import os

# Configuration for each model
configs = [
    {
        "label": "DQN64",
        "file": "../DQN64_1000episodes/average_DQN64_1000episodes1.txt",
        "color": "#1f77b4"  # blue
    },
    {
        "label": "DQN64x2",
        "file": "../DQN64x2_1000episodes/average_DQN64x2_1000episodes3.txt",
        "color": "#ff7f0e"  # orange
    },
    {
        "label": "DQN128",
        "file": "../DQN128_1000episodes/average_DQN128_1000episodes1.txt",
        "color": "#2ca02c"  # green
    }
]

base_dir = "Game"
plt.figure(figsize=(10, 6))

for cfg in configs:
    path = os.path.join(base_dir, cfg["file"])
    if os.path.exists(path):
        with open(path, "r") as f:
            rewards = [float(line.strip()) for line in f if line.strip()]
        plt.plot(rewards, label=cfg["label"], color=cfg["color"])
    else:
        print(f"File not found: {path}")

# Horizontal line for perfect solution
plt.axhline(y=29, color='green', linestyle='--', linewidth=2, label='Perfect Solution (29)')
# Horizontal line for Random agent
plt.axhline(y=14.73, color='gray', linestyle=':', linewidth=2, label='Random Agent (14.73)')

plt.xlabel('Evaluation Step')
plt.ylabel('Average Reward')
plt.title('DQN Comparison')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
