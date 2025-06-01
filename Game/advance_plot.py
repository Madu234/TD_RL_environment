import os
import sys
import matplotlib.pyplot as plt
import numpy as np

def process_txt_files(folder):
    combined_results = []

    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        if os.path.isdir(subfolder_path) and "episodes" in subfolder.lower():
            if "random" in subfolder.lower():
                combined_results.append((subfolder, subfolder_path, True))
            else:
                combined_results.append((subfolder, subfolder_path, False))

    if combined_results:
        plot_combined_results(combined_results)

def process_file(folder, is_random):
    txt_files = [f for f in os.listdir(folder) if f.startswith('average_') and f.endswith('.txt')]
    if not txt_files:
        return

    txt_file = os.path.join(folder, txt_files[0])
    with open(txt_file, 'r') as file:
        data = [float(line.strip()) for line in file if line.strip()]

    # Calculate the average of the data
    overall_avg = np.mean(data)

    plt.figure(figsize=(10, 6))
    if is_random:
        # Plot random agent results
        plt.plot(data, label=f'RandomAgent Results', color='gray', linewidth=2, alpha=0.25)
        plt.axhline(y=overall_avg, color='gray', linestyle='-', linewidth=2, alpha=0.5, label=f'RandomAgent Avg ({overall_avg:.2f})')
    else:
        # Plot normal agent results
        plt.plot(data, label=f'NormalAgent Results', color='blue', linewidth=2, alpha=0.4)

    # Add grid, labels, and title
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlabel('Time (1 unit = 10 episodes)', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.title(f'Advanced Rewards Plot: {os.path.basename(folder)}', fontsize=14)
    plt.legend(loc='upper left', fontsize=10)

    # Save the plot
    output_folder = os.path.join(folder, 'advanced_plots')
    os.makedirs(output_folder, exist_ok=True)
    output_filename = os.path.join(output_folder, f"{os.path.basename(folder)}.jpg")
    plt.savefig(output_filename, format='jpg')
    plt.close()
    print(f"Advanced plot saved to {output_filename}")

def plot_combined_results(results):
    plt.figure(figsize=(12, 8))
    # First plot DQN results, then random agent results
    dqn_results = [r for r in results if not r[2]]
    random_results = [r for r in results if r[2]]

    # Plot DQN results first
    for subfolder, folder_path, _ in dqn_results:
        txt_files = [f for f in os.listdir(folder_path) if f.startswith('average_') and f.endswith('.txt')]
        if not txt_files:
            continue
        txt_file = os.path.join(folder_path, txt_files[0])
        with open(txt_file, 'r') as file:
            data = [float(line.strip()) for line in file if line.strip()]
        # Use moving average of last 10 points (each point = 10 episodes)
        avg_10 = [np.mean(data[max(0, i-9):i+1]) for i in range(len(data))]
        x = np.arange(len(avg_10))
        z = np.polyfit(x, avg_10, 1)
        p = np.poly1d(z)
        plt.plot(avg_10, label=f'{subfolder} Avg (Last 100 Episodes)', color='blue', linewidth=2, alpha=0.7)
        plt.plot(x, p(x), linestyle='--', color='red', linewidth=2, label=f'{subfolder} Trendline')
        # Avg of last 300 episodes (last 30 points)
        if len(avg_10) >= 30:
            avg_last_30 = np.mean(avg_10[-30:])
            plt.axhline(y=avg_last_30, color='blue', linestyle=':', linewidth=2, alpha=0.7, label=f'{subfolder} Last 300 Avg ({avg_last_30:.2f})')
        else:
            avg_last_30 = np.mean(avg_10)
            plt.axhline(y=avg_last_30, color='blue', linestyle=':', linewidth=2, alpha=0.7, label=f'{subfolder} Last N Avg ({avg_last_30:.2f})')

    # Then plot random agent results over DQN
    for subfolder, folder_path, _ in random_results:
        txt_files = [f for f in os.listdir(folder_path) if f.startswith('average_') and f.endswith('.txt')]
        if not txt_files:
            continue
        txt_file = os.path.join(folder_path, txt_files[0])
        with open(txt_file, 'r') as file:
            data = [float(line.strip()) for line in file if line.strip()]
        overall_avg = np.mean(data)
        #plt.plot(data, label=f'{subfolder} Results', color='gray', linewidth=2, alpha=0.25)
        plt.axhline(y=overall_avg, color='gray', linestyle='-', linewidth=2, alpha=0.5, label=f'{subfolder} Avg ({overall_avg:.2f})')

    # Add green line for the maximum possible outcome
    plt.axhline(y=29, color='green', linestyle='-', linewidth=2, alpha=0.8, label='Perfect solution (29 reward)')

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlabel('Time (1 unit = 10 episodes)', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.title('Methods results', fontsize=16)
    plt.legend(loc='upper left', fontsize=10)

    output_filename = os.path.join(folder, 'Results.jpg')
    plt.savefig(output_filename, format='jpg')
    plt.close()
    print(f"Combined plot saved to {output_filename}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 advance_plot.py <Game folder>")
        sys.exit(1)

    folder = sys.argv[1]
    if not os.path.exists(folder):
        print(f"Error: Folder '{folder}' does not exist.")
        sys.exit(1)

    process_txt_files(folder)